from __future__ import print_function
# from trainer_Base import *

import os, pdb
import StringIO
import scipy.misc
import numpy as np
import glob
from itertools import chain
from collections import deque
import pickle, shutil
from tqdm import tqdm
from tqdm import trange

from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
from PIL import Image
from tensorflow.python.ops import control_flow_ops, sparse_ops

import models
from utils import *
import tflib as lib
from wgan_gp_128x64 import *
from datasets import market1501, deepfashion, dataset_utils

##############################################################################################
######################### Market1501 with FgBgPose BodyROIVis ################################
class DPIG_Encoder_GAN_BodyROI(object):
    def __init__(self, config):
        self._common_init(config)

        self.keypoint_num = 18
        self.D_arch = config.D_arch
        self.part_num = 37  ## Also change *7 --> *37 in datasets/market1501.py
        if 'market' in config.dataset.lower():
            if config.is_train:
                self.dataset_obj = market1501.get_split('train', config.data_path)
            else:
                self.dataset_obj = market1501.get_split('test', config.data_path)

        self.x, self.x_target, self.pose, self.pose_target, self.pose_rcv, self.pose_rcv_target, self.mask_r4, self.mask_r4_target, self.mask_r6, self.mask_r6_target, \
                self.part_bbox, self.part_bbox_target, self.part_vis, self.part_vis_target = self._load_batch_pair_pose(self.dataset_obj)
                
    def _common_init(self, config):
        self.config = config
        self.data_loader = None
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(config.start_step, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, dtype=tf.float32, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, dtype=tf.float32, name='d_lr')
        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.img_H, self.img_W = config.img_H, config.img_W

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = self._get_conv_shape()
        self.repeat_num = int(np.log2(height)) - 2

        self.data_path = config.data_path
        self.pretrained_path = config.pretrained_path
        self.pretrained_appSample_path = config.pretrained_appSample_path
        self.pretrained_poseAE_path = config.pretrained_poseAE_path
        self.pretrained_poseSample_path = config.pretrained_poseSample_path
        self.ckpt_path = config.ckpt_path
        self.z_emb_dir = config.z_emb_dir
        self.start_step = config.start_step
        self.log_step = config.log_step
        self.max_step = config.max_step
        # self.save_model_secs = config.save_model_secs
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.sample_app = config.sample_app
        self.sample_fg = config.sample_fg
        self.sample_bg = config.sample_bg
        self.sample_pose = config.sample_pose
        self.one_app_per_batch = config.one_app_per_batch
        self.interpolate_fg = config.interpolate_fg
        self.interpolate_fg_up = config.interpolate_fg_up
        self.interpolate_fg_down = config.interpolate_fg_down
        self.interpolate_bg = config.interpolate_bg
        self.interpolate_pose = config.interpolate_pose
        self.inverse_fg = config.inverse_fg
        self.inverse_bg = config.inverse_bg
        self.inverse_pose = config.inverse_pose
        self.config = config
        if self.is_train:
            self.num_threads = 4
            self.capacityCoff = 2
        else: # during testing to keep the order of the input data
            self.num_threads = 1
            self.capacityCoff = 1

    def _get_conv_shape(self):
        shape = [self.batch_size, self.img_H, self.img_W, 3]
        return shape

    def _getOptimizer(self, wgan_gp, gen_cost, disc_cost, G_var, D_var):
        clip_disc_weights = None
        if wgan_gp.MODE == 'wgan':
            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.g_lr).minimize(gen_cost,
                                                 var_list=G_var, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.d_lr).minimize(disc_cost,
                                                 var_list=D_var, colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif wgan_gp.MODE == 'wgan-gp':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                              var_list=G_var, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'dcgan':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5).minimize(gen_cost,
                                              var_list=G_var, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'lsgan':
            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.g_lr).minimize(gen_cost,
                                                 var_list=G_var, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.d_lr).minimize(disc_cost,
                                                  var_list=D_var, colocate_gradients_with_ops=True)
        else:
            raise Exception()
        return gen_train_op, disc_train_op, clip_disc_weights

    def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
        if 'DCGAN'==arch:
            return wgan_gp.DCGANDiscriminator
        elif 'FCDis'==arch:
            return wgan_gp.FCDiscriminator
        if arch.startswith('DCGANRegion'):
            return wgan_gp.DCGANDiscriminatoRegion
        raise Exception('You must choose an architecture!')
    # def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
    #     if 'Patch70x70'==arch:
    #         return wgan_gp.PatchDiscriminator_70x70
    #     elif 'Patch46x46'==arch:
    #         return wgan_gp.PatchDiscriminator_46x46
    #     elif 'Patch28x28'==arch:
    #         return wgan_gp.PatchDiscriminator_28x28
    #     elif 'Patch16x16'==arch:
    #         return wgan_gp.PatchDiscriminator_16x16
    #     elif 'Patch13x13'==arch:
    #         return wgan_gp.PatchDiscriminator_13x13
    #     elif 'DCGAN'==arch:
    #         # Baseline (G: DCGAN, D: DCGAN)
    #         return wgan_gp.DCGANDiscriminator
    #     elif 'FCDis'==arch:
    #         return wgan_gp.FCDiscriminator
    #     raise Exception('You must choose an architecture!')

    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            self.saverPart = tf.train.Saver(var1+var2, max_to_keep=20)
            
        if self.pretrained_poseAE_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseAE')
            self.saverPoseAEPart = tf.train.Saver(var, max_to_keep=20)

        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.pretrained_path is not None:
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('restored from pretrained_path:', self.pretrained_path)
        if self.pretrained_poseAE_path is not None:
            self.saverPoseAEPart.restore(self.sess, self.pretrained_poseAE_path)
            print('restored from pretrained_poseAE_path:', self.pretrained_poseAE_path)
        if self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

        self.test_dir_name = 'test_result'

    def _gan_loss(self, wgan_gp, Discriminator, disc_real, disc_fake, real_data=None, fake_data=None, arch=None):
        if wgan_gp.MODE == 'wgan':
            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        elif wgan_gp.MODE == 'wgan-gp':
            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[wgan_gp.BATCH_SIZE/len(wgan_gp.DEVICES),1,1,1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (tf.squeeze(alpha,[2,3])*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += wgan_gp.LAMBDA*gradient_penalty

        elif wgan_gp.MODE == 'dcgan':
            gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                              labels=tf.ones_like(disc_fake)))
            disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                labels=tf.zeros_like(disc_fake)))
            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                labels=tf.ones_like(disc_real)))                                     
            disc_cost /= 2.
        elif wgan_gp.MODE == 'lsgan':
            gen_cost = tf.reduce_mean((disc_fake - 1)**2)
            disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

        else:
            raise Exception()
        return gen_cost, disc_cost

    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.img_H*self.img_W*3)
        self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

    def build_model(self):
        self._define_input()
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-3 (totally 3)
            # self.embs, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI(self.x, self.part_bbox, len(indices), 64, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
            # Part 1-7 (totally 7)
            indices = range(7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI(self.x, select_part_bbox, len(indices), 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            # self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI2(self.x, select_part_bbox, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=0.9, reuse=False)
            ## Part 1,4-8 (totally 6)
            # indices = [1] + range(4,9)
            # select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            # self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI(self.x, select_part_bbox, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            # select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            # select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            # self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROIVis(self.x, select_part_bbox, select_part_vis, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, self.G_var = self.Generator_fn(
                    self.embs_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G_var += self.Encoder_var

        self.G = denorm_img(G, self.data_format)
        pair = tf.concat([self.x, G], 0)
        self.D_z = self.Discriminator_fn(tf.transpose( pair, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')
        D_z_pos, D_z_neg = tf.split(self.D_z, 2)

        self.g_loss, self.d_loss = self._gan_loss(self.wgan_gp, self.Discriminator_fn, D_z_pos, D_z_neg, arch=self.D_arch)
        self.PoseMaskLoss = tf.reduce_mean(tf.abs(G - self.x) * (self.mask_r6))
        self.L1Loss = tf.reduce_mean(tf.abs(G - self.x))
        self.g_loss_only = self.g_loss

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/PoseMaskLoss", self.PoseMaskLoss),
            tf.summary.scalar("loss/L1Loss", self.L1Loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_only", self.g_loss_only),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _define_loss_optim(self):
        self.g_loss += self.L1Loss * 20
        self.g_optim, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss, self.d_loss, self.G_var, self.D_var)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed, part_bbox_fixed, \
                    part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step>0:
                self.sess.run([self.g_optim])
            # Train critic
            if (self.wgan_gp.MODE == 'dcgan') or (self.wgan_gp.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run(self.d_optim)
                if self.wgan_gp.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights)

            if 0==step or step % self.log_step == self.log_step-1:
                fetch_dict = {
                    "summary": self.summary_op
                }
                result = self.sess.run(fetch_dict)
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or step % (self.log_step * 3) == (self.log_step * 3)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, part_bbox_fixed, part_vis_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def test(self):
        test_result_dir = os.path.join(self.model_dir, 'test_result')
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for i in xrange(100):
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed, part_bbox_fixed, \
                        part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if 0==i:
                x_fake = self.generate(x, x_target, pose_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=self.start_step, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=self.start_step, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, idx))
                im = Image.fromarray(p[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                im = Image.fromarray(mask_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask, idx))
                im = Image.fromarray(mask_target_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask_target, idx))
            if 0==i:
                save_image(x_fixed, '{}/x_fixed.png'.format(test_result_dir))
                save_image(x_target_fixed, '{}/x_target_fixed.png'.format(test_result_dir))
                save_image(mask_fixed, '{}/mask_fixed.png'.format(test_result_dir))
                save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(test_result_dir))

    def test_one_by_one(self, img_dir, pair_path, all_peaks_path, subsets_path,
                    pair_num=500, shuffle=True, random_int=0, result_dir_name='test_demo'):
        test_result_dir = os.path.join(self.model_dir, result_dir_name)
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        pairs = pickle.load(open(pair_path,'r'))
        all_peaks_dic = pickle.load(open(all_peaks_path,'r'))
        subsets_dic = pickle.load(open(subsets_path,'r'))

        if shuffle:
            np.random.seed(0)
            idx_all = np.random.permutation(len(pairs))
        else:
            idx_all = np.array(range(len(pairs)))
        # idx_list = idx_all[:test_pair_num]
        height, width, _ = scipy.misc.imread(os.path.join(img_dir, pairs[0][0])).shape

        cnt = -1
        for i in trange(len(idx_all)):
            if cnt>= pair_num-1:
                break
            idx = idx_all[i]
            
            if (pairs[idx][0] in all_peaks_dic) and (pairs[idx][1] in all_peaks_dic):
                cnt += 1
                ## Pose 0
                peaks_0 = _get_valid_peaks(all_peaks_dic[pairs[idx][0]], subsets_dic[pairs[idx][0]])
                indices_r4_0, values_r4_0, shape = _getSparsePose(peaks_0, height, width, self.keypoint_num, radius=4, mode='Solid')
                pose_dense_0 = _sparse2dense(indices_r4_0, values_r4_0, shape)
                pose_mask_r4_0 = _getPoseMask(peaks_0, height, width, radius=4, mode='Solid')
                ## Pose 1
                peaks_1 = _get_valid_peaks(all_peaks_dic[pairs[idx][1]], subsets_dic[pairs[idx][1]])
                indices_r4_1, values_r4_1, shape = _getSparsePose(peaks_1, height, width, self.keypoint_num, radius=4, mode='Solid')
                pose_dense_1 = _sparse2dense(indices_r4_1, values_r4_1, shape)
                pose_mask_r4_1 = _getPoseMask(peaks_1, height, width, radius=4, mode='Solid')
                ## Generate image
                x = scipy.misc.imread(os.path.join(img_dir, pairs[idx][0]))
                x = process_image(x, 127.5, 127.5)
                x_target = scipy.misc.imread(os.path.join(img_dir, pairs[idx][1]))
                x_target = process_image(x_target, 127.5, 127.5)
                x_batch = np.expand_dims(x,axis=0)
                pose_batch = np.expand_dims(pose_dense_1*2-1,axis=0)
                G = self.sess.run(self.G, {self.x: x_batch, self.pose_target: pose_batch})
                ## Save
                shutil.copy(os.path.join(img_dir, pairs[idx][0]), os.path.join(test_result_dir_x, 'pair%05d-%s'%(cnt, pairs[idx][0])))
                shutil.copy(os.path.join(img_dir, pairs[idx][1]), os.path.join(test_result_dir_x_target, 'pair%05d-%s'%(cnt, pairs[idx][1])))
                im = Image.fromarray(G.squeeze().astype(np.uint8))
                im.save('%s/pair%05d-%s-%s.jpg'%(test_result_dir_G, cnt, pairs[idx][0], pairs[idx][1]))
                im = np.amax(pose_dense_0, axis=-1, keepdims=False)*255
                im = Image.fromarray(im.astype(np.uint8))
                im.save('%s/pair%05d-%s.jpg'%(test_result_dir_pose, cnt, pairs[idx][0]))
                im = np.amax(pose_dense_1, axis=-1, keepdims=False)*255
                im = Image.fromarray(im.astype(np.uint8))
                im.save('%s/pair%05d-%s.jpg'%(test_result_dir_pose_target, cnt, pairs[idx][1]))
                im = pose_mask_r4_0*255
                im = Image.fromarray(im.astype(np.uint8))
                im.save('%s/pair%05d-%s.jpg'%(test_result_dir_mask, cnt, pairs[idx][0]))
                im = pose_mask_r4_1*255
                im = Image.fromarray(im.astype(np.uint8))
                im.save('%s/pair%05d-%s.jpg'%(test_result_dir_mask_target, cnt, pairs[idx][1]))
            else:
                continue

    def generate(self, x_fixed, x_target_fixed, pose_fixed, part_bbox_fixed, part_vis_fixed, root_path=None, path=None, idx=None, save=True):
        G = self.sess.run(self.G, {self.x: x_fixed, self.pose: pose_fixed, self.part_bbox: part_bbox_fixed, self.part_vis: part_vis_fixed})
        ssim_G_x_list = []
        for i in xrange(G.shape[0]):
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_gray = rgb2gray(((x_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_gray, data_range=x_gray.max() - x_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
        return G

    def get_image_from_loader(self):
        x, x_target, pose, pose_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target = self.sess.run([self.x, self.x_target, self.pose, self.pose_target, 
                                        self.mask_r6, self.mask_r6_target, self.part_bbox, self.part_bbox_target, self.part_vis, self.part_vis_target])
        x = unprocess_image(x, 127.5, 127.5)
        x_target = unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target

    def _load_batch_pair_pose(self, dataset):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        image_raw_0, image_raw_1, label, pose_rcv_0, pose_rcv_1, mask_r4_0, mask_r4_1, mask_r6_0, mask_r6_1, part_bbox_0, part_bbox_1, part_vis_0, part_vis_1  = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_peaks_0_rcv', 'pose_peaks_1_rcv', 'pose_mask_r4_0', 'pose_mask_r4_1', 'pose_mask_r6_0', 'pose_mask_r6_1', 
            'part_bbox_0', 'part_bbox_1', 'part_vis_0', 'part_vis_1'])

        image_raw_0 = tf.reshape(image_raw_0, [128, 64, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [128, 64, 3]) 
        mask_r4_0 = tf.cast(tf.reshape(mask_r4_0, [128, 64, 1]), tf.float32)
        mask_r4_1 = tf.cast(tf.reshape(mask_r4_1, [128, 64, 1]), tf.float32)
        mask_r6_0 = tf.cast(tf.reshape(mask_r6_0, [128, 64, 1]), tf.float32)
        mask_r6_1 = tf.cast(tf.reshape(mask_r6_1, [128, 64, 1]), tf.float32)
        part_bbox_0 = tf.reshape(part_bbox_0, [self.part_num, 4])        
        part_bbox_1 = tf.reshape(part_bbox_1, [self.part_num, 4]) 

        images_0, images_1, poses_rcv_0, poses_rcv_1, masks_r4_0, masks_r4_1, masks_r6_0, masks_r6_1, part_bboxs_0, part_bboxs_1, part_viss_0, part_viss_1 = tf.train.batch([image_raw_0, 
                                                    image_raw_1, pose_rcv_0, pose_rcv_1, mask_r4_0, mask_r4_1, mask_r6_0, mask_r6_1, part_bbox_0, part_bbox_1, part_vis_0, part_vis_1], 
                                                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = tf.cast(coord2channel_simple_rcv(poses_rcv_0, keypoint_num=18, is_normalized=False, img_H=128, img_W=64), tf.float32)
        poses_1 = tf.cast(coord2channel_simple_rcv(poses_rcv_1, keypoint_num=18, is_normalized=False, img_H=128, img_W=64), tf.float32)
        poses_0 = tf_poseInflate(poses_0, keypoint_num=18, radius=4, img_H=self.img_H, img_W=self.img_W)
        poses_1 = tf_poseInflate(poses_1, keypoint_num=18, radius=4, img_H=self.img_H, img_W=self.img_W)

        return images_0, images_1, poses_0, poses_1, poses_rcv_0, poses_rcv_1, masks_r4_0, masks_r4_1, masks_r6_0, masks_r6_1, part_bboxs_0, part_bboxs_1, part_viss_0, part_viss_1


class DPIG_Encoder_GAN_BodyROI_FgBg(DPIG_Encoder_GAN_BodyROI):
    def build_model(self):
        self._define_input()
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1,8-16 (totally 10)
            # indices = [1] + range(8,17)
            ## Part 1-7 (totally 7)
            indices = range(0,7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            # self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaOneBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            self.embs, _, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            # self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgImgOneBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            # self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgImgTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, self.G_var = self.Generator_fn(
                    self.embs_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        self.G_var += self.Encoder_var

        self.G = denorm_img(G, self.data_format)

        D_z_pos = self.Discriminator_fn(tf.transpose( self.x, [0,3,1,2] ), input_dim=3)
        D_z_neg = self.Discriminator_fn(tf.transpose( G, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')

        self.g_loss, self.d_loss = self._gan_loss(self.wgan_gp, self.Discriminator_fn, D_z_pos, D_z_neg, arch=self.D_arch)
        self.PoseMaskLoss = tf.reduce_mean(tf.abs(G - self.x) * (self.mask_r6))
        self.L1Loss = tf.reduce_mean(tf.abs(G - self.x))
        self.g_loss_only = self.g_loss

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/PoseMaskLoss", self.PoseMaskLoss),
            tf.summary.scalar("loss/L1Loss", self.L1Loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_only", self.g_loss_only),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _define_loss_optim(self):
        self.g_loss += self.L1Loss * 20
        self.g_optim, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss, self.d_loss, self.G_var, self.D_var)



class DPIG_PoseRCV_AE_BodyROI(DPIG_Encoder_GAN_BodyROI):
    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.keypoint_num*3)
        self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)
            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=32, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)

            if self.sample_pose: ## Sampling new poses during testing
                self.pose_embs = tf.random_normal(tf.shape(self.pose_embs), mean=0.0, stddev=0.2, dtype=tf.float32)

            G_pose_coord, G_pose_visible, self.G_var_decoder = models.PoseDecoderFCRes(self.pose_embs, self.keypoint_num, 
                        repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            self.G_var_pose = self.G_var_encoder + self.G_var_decoder

        G_pose_rcv = tf.concat([tf.reshape(G_pose_coord, [self.batch_size,self.keypoint_num,2]), tf.expand_dims(G_pose_visible,-1)], axis=-1)
        G_pose = coord2channel_simple_rcv(G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
        self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)
        self.G_pose_rcv = G_pose_rcv

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - G_pose_rcv))

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/reconstruct_loss", self.reconstruct_loss),
        ])

    def _define_loss_optim(self):
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5).minimize(self.reconstruct_loss * 20,
                                              var_list=self.G_var_pose, colocate_gradients_with_ops=True)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, mask_fixed, mask_target_fixed, \
                                        part_bbox_fixed, part_bbox_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step>0:
                self.sess.run([self.g_optim])

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def get_image_from_loader(self):
        x, x_target, pose, pose_target, pose_rcv, mask, mask_target, part_bbox, part_bbox_target = self.sess.run([self.x, self.x_target, self.pose, self.pose_target, 
                                self.pose_rcv, self.mask_r6, self.mask_r6_target, self.part_bbox, self.part_bbox_target])
        x = unprocess_image(x, 127.5, 127.5)
        x_target = unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, pose_rcv, mask, mask_target, part_bbox, part_bbox_target


################### Subnet of Appearance/Pose Sampling #################
class DPIG_Encoder_subSampleAppNetFgBg_GAN_BodyROI(DPIG_Encoder_GAN_BodyROI):
    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp_fg = WGAN_GP(DATA_DIR='', MODE='wgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10)
        self.Discriminator_fg_fn = self._getDiscriminator(self.wgan_gp_fg, arch='FCDis')
        self.wgan_gp_bg = WGAN_GP(DATA_DIR='', MODE='wgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10)
        self.Discriminator_bg_fn = self._getDiscriminator(self.wgan_gp_bg, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            ## Part 1-7 (totally 7)
            indices = range(0,7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)

            self.fg_embs = tf.slice(self.embs, [0,0], [-1,len(indices)*32])
            self.bg_embs = tf.slice(self.embs, [0,len(indices)*32], [-1,-1])
        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, self.G_var = self.Generator_fn(
                    self.embs_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        with tf.variable_scope("Gaussian_FC_Fg") as vs:
            embs_shape = self.fg_embs.get_shape().as_list()
            self.app_embs_fg, self.G_var_app_embs_fg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        with tf.variable_scope("Gaussian_FC_Bg") as vs:
            embs_shape = self.bg_embs.get_shape().as_list()
            self.app_embs_bg, self.G_var_app_embs_bg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=256, data_format=self.data_format, activation_fn=LeakyReLU)

        ## Adversarial for Gaussian Fg
        # encode_pair_fg = tf.concat([self.fg_embs, self.app_embs_fg], 0)
        # D_z_embs_fg = self.Discriminator_fg_fn(encode_pair_fg, input_dim=encode_pair_fg.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Fg_FCDis_')
        # D_z_pos_embs_fg, D_z_neg_embs_fg = tf.split(D_z_embs_fg, 2)
        D_z_pos_embs_fg = self.Discriminator_fg_fn(self.fg_embs, input_dim=self.fg_embs.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Fg_FCDis_')
        D_z_neg_embs_fg = self.Discriminator_fg_fn(self.app_embs_fg, input_dim=self.app_embs_fg.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Fg_FCDis_')
        self.D_var_embs_fg = lib.params_with_name('Fg_FCDis_Discriminator.')
        self.g_loss_embs_fg, self.d_loss_embs_fg = self._gan_loss(self.wgan_gp_fg, self.Discriminator_fg_fn, 
                                            D_z_pos_embs_fg, D_z_neg_embs_fg, self.fg_embs, self.app_embs_fg)
        ## Adversarial for Gaussian Bg
        # encode_pair_bg = tf.concat([self.bg_embs, self.app_embs_bg], 0)
        # D_z_embs_bg = self.Discriminator_bg_fn(encode_pair_bg, input_dim=encode_pair_bg.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Bg_FCDis_')
        # D_z_pos_embs_bg, D_z_neg_embs_bg = tf.split(D_z_embs_bg, 2)
        D_z_pos_embs_bg = self.Discriminator_bg_fn(self.bg_embs, input_dim=self.bg_embs.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Bg_FCDis_')
        D_z_neg_embs_bg = self.Discriminator_bg_fn(self.app_embs_bg, input_dim=self.app_embs_bg.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Bg_FCDis_')
        self.D_var_embs_bg = lib.params_with_name('Bg_FCDis_Discriminator.')
        self.g_loss_embs_bg, self.d_loss_embs_bg = self._gan_loss(self.wgan_gp_bg, self.Discriminator_bg_fn, 
                                            D_z_pos_embs_bg, D_z_neg_embs_bg, self.bg_embs, self.app_embs_bg)

        assert (self.batch_size>0)and(self.batch_size%2==0), 'batch should be Even and >0'
        self.app_embs_fixFg = tf.tile(tf.slice(self.app_embs_fg, [0,0], [1,-1]), [self.batch_size/2,1])
        self.app_embs_varyFg = tf.slice(self.app_embs_fg, [self.batch_size/2,0], [self.batch_size/2,-1])
        self.app_embs_fixBg = tf.tile(tf.slice(self.app_embs_bg, [0,0], [1,-1]), [self.batch_size/2,1])
        self.app_embs_varyBg = tf.slice(self.app_embs_bg, [self.batch_size/2,0], [self.batch_size/2,-1])
        self.app_embs = tf.concat([tf.concat([self.app_embs_fixFg,self.app_embs_varyFg],axis=0), tf.concat([self.app_embs_varyBg,self.app_embs_fixBg],axis=0)], axis=-1)
        self.embs_app_rep = tf.tile(tf.expand_dims(self.app_embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_app_rep = tf.reshape(self.embs_app_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_app_rep = nchw_to_nhwc(self.embs_app_rep)
        with tf.variable_scope("ID_AE") as vs:
            # pdb.set_trace()
            G_gaussian_app, _, _ = self.Generator_fn(
                    self.embs_app_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=True)
        self.G = denorm_img(G_gaussian_app, self.data_format)

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/g_loss_embs_fg", self.g_loss_embs_fg),
            tf.summary.scalar("loss/d_loss_embs_fg", self.d_loss_embs_fg),
            tf.summary.scalar("loss/g_loss_embs_bg", self.g_loss_embs_bg),
            tf.summary.scalar("loss/d_loss_embs_bg", self.d_loss_embs_bg),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _define_loss_optim(self):
        self.g_optim_embs_fg, self.d_optim_embs_fg, self.clip_disc_weights_embs_fg = self._getOptimizer(self.wgan_gp_fg, 
                                self.g_loss_embs_fg, self.d_loss_embs_fg, self.G_var_app_embs_fg, self.D_var_embs_fg)
        self.g_optim_embs_bg, self.d_optim_embs_bg, self.clip_disc_weights_embs_bg = self._getOptimizer(self.wgan_gp_bg, 
                                self.g_loss_embs_bg, self.d_loss_embs_bg, self.G_var_app_embs_bg, self.D_var_embs_bg)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed, part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            ## Fg
            if step>0:
                self.sess.run([self.g_optim_embs_fg])
            # Train critic
            if (self.wgan_gp_fg.MODE == 'dcgan') or (self.wgan_gp_fg.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp_fg.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run(self.d_optim_embs_fg)
                if self.wgan_gp_fg.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights_embs_fg)
            ## Bg
            if step>0:
                self.sess.run([self.g_optim_embs_bg])
            # Train critic
            if (self.wgan_gp_bg.MODE == 'dcgan') or (self.wgan_gp_bg.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp_bg.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run(self.d_optim_embs_bg)
                if self.wgan_gp_bg.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights_embs_bg)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or step % (self.log_step * 1.5) == (self.log_step * 1.5)-1 or step % (self.log_step * 5) == (self.log_step * 5)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, part_bbox_fixed, part_vis_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 5) == (self.log_step * 5)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)


class DPIG_subnetSamplePoseRCV_GAN_BodyROI(DPIG_PoseRCV_AE_BodyROI):
    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp_encoder = WGAN_GP(DATA_DIR='', MODE='wgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.keypoint_num*3)
        self.Discriminator_encoder_fn = self._getDiscriminator(self.wgan_gp_encoder, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)

            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=32, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)

        embs_shape = self.pose_embs.get_shape().as_list()
        # with tf.variable_scope("Gaussian_FC") as vs:
        with tf.variable_scope("PoseGaussian") as vs:
            self.G_pose_embs, self.G_var_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        with tf.variable_scope("PoseAE") as vs:
            G_pose_coord, G_pose_visible, self.G_var_decoder = models.PoseDecoderFCRes(self.G_pose_embs, self.keypoint_num, 
                        repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            
        G_pose_rcv = tf.concat([tf.reshape(G_pose_coord, [self.batch_size,self.keypoint_num,2]), tf.expand_dims(G_pose_visible,-1)], axis=-1)
        G_pose = coord2channel_simple_rcv(G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
        self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)
        self.G_pose_rcv = G_pose_rcv


        ## Adversarial for pose_embs
        self.pose_embs = tf.reshape(self.pose_embs, [self.batch_size,-1])
        self.G_pose_embs = tf.reshape(self.G_pose_embs, [self.batch_size,-1])
        encode_pair = tf.concat([self.pose_embs, self.G_pose_embs], 0)
        self.D_z_embs = self.Discriminator_encoder_fn(encode_pair, input_dim=encode_pair.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Pose_emb_')
        self.D_var_embs = lib.params_with_name('Pose_emb_Discriminator.')
        D_z_pos, D_z_neg = tf.split(self.D_z_embs, 2)
        self.g_loss_embs, self.d_loss_embs = self._gan_loss(self.wgan_gp_encoder, self.Discriminator_encoder_fn, 
                                            D_z_pos, D_z_neg, self.pose_embs, self.G_pose_embs)

        ## Use the pose to generate person with pretrained generator
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1,1-7 (totally 7)
            indices = range(7)
            ## Part 1,8-16 (totally 10)
            # indices = [1] + range(8,17)
            # indices = [0] + range(7,16)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            # pdb.set_trace()
            self.embs, _, _, _ = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            # self.embs, _, _ = GeneratorCNN_ID_Encoder_BodyROIVis(self.x, select_part_bbox, select_part_vis, len(indices), 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            # self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI(self.x, self.part_bbox, 7, 32, 
            #                                 self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        ## Use py code to get G_pose_inflated, so the op is out of the graph
        self.G_pose_inflated = tf.placeholder(tf.float32, shape=G_pose.get_shape())
        with tf.variable_scope("ID_AE") as vs:
            G, _, _ = self.Generator_fn(
                    self.embs_rep, self.G_pose_inflated, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G = denorm_img(G, self.data_format)


        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G_pose", self.G_pose),
            tf.summary.scalar("loss/g_loss_embs", self.g_loss_embs),
            tf.summary.scalar("loss/d_loss_embs", self.d_loss_embs),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.histogram("distribution/pose_emb", self.pose_embs),
            tf.summary.histogram("distribution/G_pose_embs", self.G_pose_embs),
            tf.summary.histogram("distribution/app_emb", self.embs),
        ])

    def _define_loss_optim(self):
        self.g_optim_embs, self.d_optim_embs, self.clip_disc_weights_embs = self._getOptimizer(self.wgan_gp_encoder, 
                                self.g_loss_embs, self.d_loss_embs, self.G_var_embs, self.D_var_embs)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, mask_fixed, mask_target_fixed, \
                                            part_bbox_fixed, part_bbox_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            # Use GAN for Pose Embedding
            if step>0:
                self.sess.run([self.g_optim_embs])
            # Train critic
            if (self.wgan_gp_encoder.MODE == 'dcgan') or (self.wgan_gp_encoder.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp_encoder.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run([self.d_optim_embs])
                if self.wgan_gp_encoder.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights_embs)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or 10==step or 200==step or step % (self.log_step * 3) == (self.log_step * 3)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, part_bbox_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def generate(self, x_fixed, x_target_fixed, pose_fixed, part_bbox_fixed, root_path=None, path=None, idx=None, save=True):
        G_pose_rcv, G_pose = self.sess.run([self.G_pose_rcv, self.G_pose])
        G_pose_inflated = py_poseInflate(G_pose_rcv, is_normalized=True, radius=4, img_H=128, img_W=64)
        G = self.sess.run(self.G, {self.x: x_fixed, self.G_pose_inflated: G_pose_inflated, self.part_bbox: part_bbox_fixed})
        G_pose_inflated_img = np.tile(np.amax((G_pose_inflated+1)*127.5, axis=-1, keepdims=True), [1,1,1,3])
        
        ssim_G_x_list = []
        for i in xrange(G.shape[0]):
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_gray = rgb2gray(((x_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_gray, data_range=x_gray.max() - x_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
            path = os.path.join(root_path, '{}_G_pose.png'.format(idx))
            save_image(G_pose, path)
            print("[*] Samples saved: {}".format(path))
            path = os.path.join(root_path, '{}_G_pose_inflated.png'.format(idx))
            save_image(G_pose_inflated_img, path)
            print("[*] Samples saved: {}".format(path))
        return G





#################################################################################################
####################################### DF train models #########################################

######################### DeepFashion with AppPose BodyROI ################################
class DPIG_Encoder_GAN_BodyROI_256(DPIG_Encoder_GAN_BodyROI):
    def __init__(self, config):
        self._common_init(config)
        self.keypoint_num = 18
        self.part_num = 37
        self.D_arch = config.D_arch
        if 'deepfashion' in config.dataset.lower() or 'df' in config.dataset.lower():
            if config.is_train:
                self.dataset_obj = deepfashion.get_split('train', config.data_path)
            else:
                self.dataset_obj = deepfashion.get_split('test', config.data_path)
        self.x, self.x_target, self.pose, self.pose_target, self.pose_rcv, self.pose_rcv_target, self.mask, self.mask_target, \
                self.part_bbox, self.part_bbox_target, self.part_vis, self.part_vis_target = self._load_batch_pair_pose(self.dataset_obj)

    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.img_H*self.img_W*3)
        self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

    def build_model(self):
        self._define_input()
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 7)
            indices = range(7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis(self.x,select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num+1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, roi_size=64, reuse=False)
            ## Part 1,4-8 (totally 6)
            # # indices = [1] + range(4,9)
            # indices = [0] + range(3,8)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            
        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, self.G_var = self.Generator_fn(
                    self.embs_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        self.G_var += self.Encoder_var

        self.G = denorm_img(G, self.data_format)

        pair = tf.concat([self.x, G], 0)

        self.D_z = self.Discriminator_fn(tf.transpose( pair, [0,3,1,2] ), input_dim=3)
        self.D_var = lib.params_with_name('Discriminator.')

        D_z_pos, D_z_neg = tf.split(self.D_z, 2)

        self.g_loss, self.d_loss = self._gan_loss(self.wgan_gp, self.Discriminator_fn, D_z_pos, D_z_neg, arch=self.D_arch)
        self.PoseMaskLoss = tf.reduce_mean(tf.abs(G - self.x) * (self.mask))
        self.L1Loss = tf.reduce_mean(tf.abs(G - self.x))
        self.g_loss_only = self.g_loss

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/PoseMaskLoss", self.PoseMaskLoss),
            tf.summary.scalar("loss/L1Loss", self.L1Loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/g_loss_only", self.g_loss_only),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _define_loss_optim(self):
        self.g_loss += self.L1Loss * 20
        self.g_optim, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss, self.d_loss, self.G_var, self.D_var)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
            part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step>0:
                self.sess.run([self.g_optim])
            # Train critic
            if (self.wgan_gp.MODE == 'dcgan') or (self.wgan_gp.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run(self.d_optim)
                if self.wgan_gp.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
                    # "k_t": self.k_t,
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or 10==step or step % (self.log_step * 3) == (self.log_step * 3)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, part_bbox_fixed, part_vis_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def test(self):
        test_result_dir = os.path.join(self.model_dir, self.test_dir_name)
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for i in xrange(100): ## for test Samples
        # for i in xrange(800): ## for IS score
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
                part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if 0==i:
                x_fake = self.generate(x, x_target, pose_fixed, part_bbox_fixed, test_result_dir, part_vis_fixed, idx=i, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_fixed, part_bbox_fixed, test_result_dir, part_vis_fixed, idx=i, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, idx))
                im = Image.fromarray(p[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                # pdb.set_trace()
                im = Image.fromarray(mask_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask, idx))
                im = Image.fromarray(mask_target_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask_target, idx))
            # pdb.set_trace()
            if 0==i:
                save_image(x_fixed, '{}/x_fixed.png'.format(test_result_dir))
                save_image(x_target_fixed, '{}/x_target_fixed.png'.format(test_result_dir))
                save_image(mask_fixed, '{}/mask_fixed.png'.format(test_result_dir))
                save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(test_result_dir))

    def generate(self, x_fixed, x_target_fixed, pose_fixed, part_bbox_fixed, part_vis_fixed, root_path=None, path=None, idx=None, save=True):
        G = self.sess.run(self.G, {self.x: x_fixed, self.pose: pose_fixed, self.part_bbox: part_bbox_fixed, self.part_vis: part_vis_fixed})
        ssim_G_x_list = []
        for i in xrange(G.shape[0]):
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_gray = rgb2gray(((x_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_gray, data_range=x_gray.max() - x_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
        return G

    # def generate(self, x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, part_bbox_fixed, root_path=None, path=None, idx=None, save=True):
    #     G = self.sess.run(self.G, {self.x: x_fixed, self.pose: pose_fixed, self.pose_target: pose_target_fixed, self.part_bbox: part_bbox_fixed})
    #     ssim_G_x_list = []
    #     for i in xrange(G.shape[0]):
    #         G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
    #         x_gray = rgb2gray(((x_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
    #         ssim_G_x_list.append(ssim(G_gray, x_gray, data_range=x_gray.max() - x_gray.min(), multichannel=False))
    #     ssim_G_x_mean = np.mean(ssim_G_x_list)
    #     if path is None and save:
    #         path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
    #         save_image(G, path)
    #         print("[*] Samples saved: {}".format(path))
    #     return G

    def get_image_from_loader(self):
        x, x_target, pose, pose_target, pose_rcv, pose_rcv_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target = self.sess.run([self.x, self.x_target, self.pose, self.pose_target, 
                        self.pose_rcv, self.pose_rcv_target, self.mask, self.mask_target, self.part_bbox, self.part_bbox_target, self.part_vis, self.part_vis_target])
        x = unprocess_image(x, 127.5, 127.5)
        x_target = unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, pose_rcv, pose_rcv_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target

    def _load_batch_pair_pose(self, dataset):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        image_raw_0, image_raw_1, label, pose_rcv_0, pose_rcv_1, mask_0, mask_1, part_bbox_0, part_bbox_1, part_vis_0, part_vis_1  = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_peaks_0_rcv', 'pose_peaks_1_rcv', 'pose_mask_r4_0', 'pose_mask_r4_1', 
            'part_bbox_0', 'part_bbox_1', 'part_vis_0', 'part_vis_1'])

        image_raw_0 = tf.reshape(image_raw_0, [256, 256, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [256, 256, 3]) 
        mask_0 = tf.cast(tf.reshape(mask_0, [256, 256, 1]), tf.float32)
        mask_1 = tf.cast(tf.reshape(mask_1, [256, 256, 1]), tf.float32)
        part_bbox_0 = tf.reshape(part_bbox_0, [self.part_num, 4])        
        part_bbox_1 = tf.reshape(part_bbox_1, [self.part_num, 4]) 

        images_0, images_1, poses_rcv_0, poses_rcv_1, masks_0, masks_1, part_bboxs_0, part_bboxs_1, part_viss_0, part_viss_1 = tf.train.batch([image_raw_0, 
                                                    image_raw_1, pose_rcv_0, pose_rcv_1, mask_0, mask_1, part_bbox_0, part_bbox_1, part_vis_0, part_vis_1], 
                                                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = tf.cast(coord2channel_simple_rcv(poses_rcv_0, keypoint_num=18, is_normalized=False, img_H=256, img_W=256), tf.float32)
        poses_1 = tf.cast(coord2channel_simple_rcv(poses_rcv_1, keypoint_num=18, is_normalized=False, img_H=256, img_W=256), tf.float32)
        poses_0 = tf_poseInflate(poses_0, keypoint_num=18, radius=4, img_H=self.img_H, img_W=self.img_W)
        poses_1 = tf_poseInflate(poses_1, keypoint_num=18, radius=4, img_H=self.img_H, img_W=self.img_W)
        
        return images_0, images_1, poses_0, poses_1, poses_rcv_0, poses_rcv_1, masks_0, masks_1, part_bboxs_0, part_bboxs_1, part_viss_0, part_viss_1



class DPIG_Encoder_subSampleAppNet_GAN_BodyROI_256(DPIG_Encoder_GAN_BodyROI_256):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            self.saverPart = tf.train.Saver(var1+var2, max_to_keep=20)
            
        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=False)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.pretrained_path is not None:
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('restored from pretrained_path:', self.pretrained_path)
        elif self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp_encoder = WGAN_GP(DATA_DIR='', MODE='wgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=7*32)
        self.Discriminator_encoder_fn = self._getDiscriminator(self.wgan_gp_encoder, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 7)
            indices = range(7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROI(self.x, select_part_bbox, len(indices), 32, 
                                            self.repeat_num+1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, self.G_var = self.Generator_fn(
                    self.embs_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        embs_shape = self.embs.get_shape().as_list()
        with tf.variable_scope("Gaussian_FC") as vs:
            self.app_embs, self.G_var_app_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        ## Adversarial for Gaussian
        encode_pair = tf.concat([self.embs, self.app_embs], 0)
        self.D_z_embs = self.Discriminator_encoder_fn(encode_pair, input_dim=encode_pair.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='FCDis_')
        self.D_var_embs = lib.params_with_name('FCDis_Discriminator.')
        D_z_pos_embs, D_z_neg_embs = tf.split(self.D_z_embs, 2)
        self.g_loss_embs, self.d_loss_embs = self._gan_loss(self.wgan_gp_encoder, self.Discriminator_encoder_fn, 
                                            D_z_pos_embs, D_z_neg_embs, self.embs, self.app_embs)

        self.embs_app_rep = tf.tile(tf.expand_dims(self.app_embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_app_rep = tf.reshape(self.embs_app_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_app_rep = nchw_to_nhwc(self.embs_app_rep)
        with tf.variable_scope("ID_AE") as vs:
            G_gaussian_app, _, _ = self.Generator_fn(
                    self.embs_app_rep, self.pose, 
                    self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=True)
        self.G = denorm_img(G_gaussian_app, self.data_format)

        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/g_loss_embs", self.g_loss_embs),
            tf.summary.scalar("loss/d_loss_embs", self.d_loss_embs),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    def _define_loss_optim(self):
        self.g_optim_embs, self.d_optim_embs, self.clip_disc_weights_embs = self._getOptimizer(self.wgan_gp_encoder, 
                                self.g_loss_embs, self.d_loss_embs, self.G_var_app_embs, self.D_var_embs)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed, part_bbox_fixed, part_bbox_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step>0:
                self.sess.run([self.g_optim_embs])
            # Train critic
            if (self.wgan_gp_encoder.MODE == 'dcgan') or (self.wgan_gp_encoder.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp_encoder.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run(self.d_optim_embs)
                if self.wgan_gp_encoder.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights_embs)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or step % (self.log_step * 3) == (self.log_step * 3)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, pose_target_fixed, part_bbox_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

class DPIG_PoseRCV_AE_BodyROI_256(DPIG_Encoder_GAN_BodyROI_256):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            self.saverPart = tf.train.Saver(var1+var2, max_to_keep=20)
            
        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.pretrained_path is not None:
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('restored from pretrained_path:', self.pretrained_path)
        elif self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.keypoint_num*3)
        self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)
            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=32, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)

            if self.sample_pose: ## Sampling new poses during testing
                self.pose_embs = tf.random_normal(tf.shape(self.pose_embs), mean=0.0, stddev=0.2, dtype=tf.float32)

            G_pose_coord, G_pose_visible, self.G_var_decoder = models.PoseDecoderFCRes(self.pose_embs, self.keypoint_num, 
                        repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)

            self.G_var_pose = self.G_var_encoder + self.G_var_decoder

        G_pose_rcv = tf.concat([tf.reshape(G_pose_coord, [self.batch_size,self.keypoint_num,2]), tf.expand_dims(G_pose_visible,-1)], axis=-1)
        self.G_pose_rcv = G_pose_rcv

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - G_pose_rcv))


        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/reconstruct_loss", self.reconstruct_loss),
        ])

    def _define_loss_optim(self):
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5).minimize(self.reconstruct_loss * 20,
                                              var_list=self.G_var_pose, colocate_gradients_with_ops=True)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
            part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        # save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        # save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            if step>0:
                self.sess.run([self.g_optim])

            if step % self.log_step == self.log_step-1:
                fetch_dict = {
                    "summary": self.summary_op,
                    "reconstruct_loss": self.reconstruct_loss
                }
                result = self.sess.run(fetch_dict)
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()
                print('reconstruct_loss:%f'%result['reconstruct_loss'])

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)



class DPIG_subnetSamplePoseRCV_GAN_BodyROI_256(DPIG_PoseRCV_AE_BodyROI_256):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            self.saverPart = tf.train.Saver(var1+var2, max_to_keep=20)
            
        if self.pretrained_poseAE_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseAE')
            self.saverPoseAEPart = tf.train.Saver(var, max_to_keep=20)

        self.saver = tf.train.Saver(max_to_keep=20)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.pretrained_path is not None:
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('restored from pretrained_path:', self.pretrained_path)
        if self.pretrained_poseAE_path is not None:
            self.saverPoseAEPart.restore(self.sess, self.pretrained_poseAE_path)
            print('restored from pretrained_poseAE_path:', self.pretrained_poseAE_path)
        if self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

    def _define_input(self):
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp_encoder = WGAN_GP(DATA_DIR='', MODE='wgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.keypoint_num*3)
        self.Discriminator_encoder_fn = self._getDiscriminator(self.wgan_gp_encoder, arch='FCDis')

    def build_model(self):
        self._define_input()
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)

            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=32, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            
        embs_shape = self.pose_embs.get_shape().as_list()
        # with tf.variable_scope("Gaussian_FC") as vs:
        with tf.variable_scope("PoseGaussian") as vs:
            self.G_pose_embs, self.G_var_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)
            # self.G_pose_embs, self.G_var_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=6, hidden_num=1024, data_format=self.data_format, activation_fn=LeakyReLU)

            
        with tf.variable_scope("PoseAE") as vs:
            G_pose_coord, G_pose_visible, self.G_var_decoder = models.PoseDecoderFCRes(self.G_pose_embs, self.keypoint_num, 
                        repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            
        G_pose_rcv = tf.concat([tf.reshape(G_pose_coord, [self.batch_size,self.keypoint_num,2]), tf.expand_dims(G_pose_visible,-1)], axis=-1)
        G_pose = coord2channel_simple_rcv(G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
        self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)
        self.G_pose_rcv = G_pose_rcv

        ## Adversarial for pose_embs
        self.pose_embs = tf.reshape(self.pose_embs, [self.batch_size,-1])
        self.G_pose_embs = tf.reshape(self.G_pose_embs, [self.batch_size,-1])
        encode_pair = tf.concat([self.pose_embs, self.G_pose_embs], 0)
        self.D_z_embs = self.Discriminator_encoder_fn(encode_pair, input_dim=encode_pair.get_shape().as_list()[-1], FC_DIM=512, n_layers=3, reuse=False, name='Pose_emb_')
        self.D_var_embs = lib.params_with_name('Pose_emb_Discriminator.')
        D_z_pos, D_z_neg = tf.split(self.D_z_embs, 2)
        self.g_loss_embs, self.d_loss_embs = self._gan_loss(self.wgan_gp_encoder, self.Discriminator_encoder_fn, 
                                            D_z_pos, D_z_neg, self.pose_embs, self.G_pose_embs)

        # ## Use the pose to generate person with pretrained generator
        # with tf.variable_scope("Encoder") as vs:
        #     pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
        #     pv_list = tf.split(self.part_vis, self.part_num, axis=1)
        #     ## Part 1,1-7 (totally 7)
        #     indices = range(7)
        #     ## Part 1,8-16 (totally 10)
        #     # indices = [1] + range(8,17)
        #     # indices = [0] + range(7,16)
        #     select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
        #     select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
        #     # pdb.set_trace()
        #     self.embs, _, _, _ = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
        #                                     self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
        #     # self.embs, _, _ = models.GeneratorCNN_ID_Encoder_BodyROIVis(self.x, select_part_bbox, select_part_vis, len(indices), 32, 
        #     #                                 self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
        #     # self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROI(self.x, self.part_bbox, 7, 32, 
        #     #                                 self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)

        # self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        # self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        # self.embs_rep = nchw_to_nhwc(self.embs_rep)

        # ## Use py code to get G_pose_inflated, so the op is out of the graph
        # self.G_pose_inflated = tf.placeholder(tf.float32, shape=G_pose.get_shape())
        # with tf.variable_scope("ID_AE") as vs:
        #     G, _, _ = self.Generator_fn(
        #             self.embs_rep, self.G_pose_inflated, 
        #             self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        # self.G = denorm_img(G, self.data_format)


        self._define_loss_optim()
        self.summary_op = tf.summary.merge([
            tf.summary.image("G_pose", self.G_pose),
            tf.summary.scalar("loss/g_loss_embs", self.g_loss_embs),
            tf.summary.scalar("loss/d_loss_embs", self.d_loss_embs),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            # tf.summary.histogram("distribution/pose_emb", self.pose_embs),
            # tf.summary.histogram("distribution/G_pose_embs", self.G_pose_embs),
            # tf.summary.histogram("distribution/app_emb", self.embs),
        ])

    def _define_loss_optim(self):
        self.g_optim_embs, self.d_optim_embs, self.clip_disc_weights_embs = self._getOptimizer(self.wgan_gp_encoder, 
                                self.g_loss_embs, self.d_loss_embs, self.G_var_embs, self.D_var_embs)

    def train(self):
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
            part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            # Use GAN for Pose Embedding
            if step>0:
                self.sess.run([self.g_optim_embs])
            # Train critic
            if (self.wgan_gp_encoder.MODE == 'dcgan') or (self.wgan_gp_encoder.MODE == 'lsgan'):
                disc_ITERS = 1
            else:
                disc_ITERS = self.wgan_gp_encoder.CRITIC_ITERS
            for i in xrange(disc_ITERS):
                self.sess.run([self.d_optim_embs])
                if self.wgan_gp_encoder.MODE == 'wgan':
                    self.sess.run(self.clip_disc_weights_embs)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if 0==step or 10==step or 200==step or step % (self.log_step * 3) == (self.log_step * 3)-1:
                x = process_image(x_fixed, 127.5, 127.5)
                x_target = process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_fixed, part_bbox_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def generate(self, x_fixed, x_target_fixed, pose_fixed, part_bbox_fixed, root_path=None, path=None, idx=None, save=True):
        G_pose_rcv, G_pose = self.sess.run([self.G_pose_rcv, self.G_pose])
        G_pose_inflated = py_poseInflate(G_pose_rcv, is_normalized=True, radius=4, img_H=256, img_W=256)
        # G = self.sess.run(self.G, {self.x: x_fixed, self.G_pose_inflated: G_pose_inflated, self.part_bbox: part_bbox_fixed})
        G_pose_inflated_img = np.tile(np.amax((G_pose_inflated+1)*127.5, axis=-1, keepdims=True), [1,1,1,3])
        
        # ssim_G_x_list = []
        # for i in xrange(G_pose.shape[0]):
        #     G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
        #     x_gray = rgb2gray(((x_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
        #     ssim_G_x_list.append(ssim(G_gray, x_gray, data_range=x_gray.max() - x_gray.min(), multichannel=False))
        # ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            # path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            # save_image(G, path)
            # print("[*] Samples saved: {}".format(path))
            path = os.path.join(root_path, '{}_G_pose.png'.format(idx))
            save_image(G_pose, path)
            print("[*] Samples saved: {}".format(path))
            path = os.path.join(root_path, '{}_G_pose_inflated.png'.format(idx))
            save_image(G_pose_inflated_img, path)
            print("[*] Samples saved: {}".format(path))
        return G_pose
