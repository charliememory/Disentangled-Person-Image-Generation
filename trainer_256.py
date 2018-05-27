from datasets import deepfashion, dataset_utils
from trainer import *
from tester import *
import scipy.misc as misc
import models

###########################################################################################
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


class DPIG_ThreeNetsApp_testOnlyCondition_256(DPIG_Encoder_GAN_BodyROI_256):
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
        if self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

        # self.test_batch_num = 30 ## for test_samples
        self.test_batch_num = 400 ## for test
        if self.sample_pose:
            self.test_dir_name = 'test_result_ROI7_Condition_TargetPose_%dx%d'%(self.test_batch_num, self.batch_size)
        else:
            self.test_dir_name = 'test_result_ROI7_Condition_TargetPose_%dx%d'%(self.test_batch_num, self.batch_size)


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
        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, _ = self.Generator_fn(
                    self.embs_rep, self.pose_target, 
                    self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G = denorm_img(G, self.data_format)


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

        for i in xrange(self.test_batch_num):
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
                part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if 0==i:
                x_fake = self.generate(x, x_target, pose_target_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_target_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, i*self.batch_size+j))
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
        
    def generate(self, x_fixed, x_target_fixed, pose_target_fixed, part_bbox_fixed, part_vis_fixed, root_path=None, path=None, idx=None, save=True):
        G = self.sess.run(self.G, {self.x: x_fixed, self.pose_target: pose_target_fixed, \
                            self.part_bbox: part_bbox_fixed, self.part_vis: part_vis_fixed})
        
        ssim_G_x_list = []
        for i in xrange(G.shape[0]):
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_target_gray = rgb2gray(((x_target_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max() - x_target_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
        return G


class DPIG_ThreeNetsApp_testOnlySampleFactor_256(DPIG_Encoder_GAN_BodyROI_256):
    def _define_input(self):
        self.train_LuNet = tf.Variable(False, name='phase')
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        # self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
        #                     LAMBDA=10, G_OUTPUT_DIM=self.img_H*self.img_W*3)
        # self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch='DCGAN')

    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            # var3 = lib.params_with_name('Discriminator.')
            self.saverPart = tf.train.Saver(var1+var2, max_to_keep=20)

        if self.pretrained_appSample_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Gaussian_FC')
            self.saverAppSamplePart = tf.train.Saver(var, max_to_keep=20)
            
        if self.pretrained_poseAE_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseAE')
            self.saverPosePart = tf.train.Saver(var, max_to_keep=20)

        if self.pretrained_poseSample_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='PoseGaussian')
            self.saverPoseSamplePart = tf.train.Saver(var, max_to_keep=20)

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
        if self.pretrained_appSample_path is not None:
            self.saverAppSamplePart.restore(self.sess, self.pretrained_appSample_path)
            print('restored from pretrained_appSample_path:', self.pretrained_appSample_path)
        if self.pretrained_poseAE_path is not None:
            self.saverPosePart.restore(self.sess, self.pretrained_poseAE_path)
            print('restored from pretrained_poseAE_path:', self.pretrained_poseAE_path)
        if self.pretrained_poseSample_path is not None:
            self.saverPoseSamplePart.restore(self.sess, self.pretrained_poseSample_path)
            print('restored from pretrained_poseSample_path:', self.pretrained_poseSample_path)
        if self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)

        self.test_batch_num = 100  ## For test
        # self.test_batch_num = 400 ## For IS score
        self.test_dir_name = 'test_result_ROI7_SampleApp%rSamplePose%r_pretrain_%dx%d'%(self.sample_app,self.sample_pose, self.test_batch_num, self.batch_size)
        # self.test_dir_name = 'test_result_ROI7_SampleFg%rSampleBg%rRealPose%r_pretrain_%dx%d'%(self.sample_fg,self.sample_bg,self.sample_pose, self.test_batch_num, self.batch_size)

    def build_model(self):
        self._define_input()
        ################################### Pose ###################################
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)
            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=32, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            pose_embs_shape = self.pose_embs.get_shape().as_list()

        with tf.variable_scope("PoseGaussian") as vs:
            self.G_pose_embs, self.G_var_embs = models.GaussianFCRes(pose_embs_shape, pose_embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)
        with tf.variable_scope("PoseAE") as vs:
            G_pose_coord, G_pose_visible, self.G_var_decoder = models.PoseDecoderFCRes(self.pose_embs, self.keypoint_num, 
                        repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
            self.G_var_pose = self.G_var_encoder + self.G_var_decoder

        if self.sample_pose: ## Sampling new poses during testing
            self.G_pose_rcv = tf.concat([tf.reshape(G_pose_coord, [self.batch_size,self.keypoint_num,2]), tf.expand_dims(G_pose_visible,-1)], axis=-1)
            G_pose = coord2channel_simple_rcv(self.G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
            self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)
        else:
            pose_rcv_norm_one = tf.tile(tf.slice(pose_rcv_norm, [0,0,0], [1,-1,-1]), [self.batch_size,1,1])
            self.G_pose_rcv = pose_rcv_norm_one
            G_pose = coord2channel_simple_rcv(self.G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
            self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - self.G_pose_rcv))

        ################################### Appearance ###################################
        ## Use the pose to generate person with pretrained generator
        self.roi_emb_dim = 32
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 7)
            indices = range(7)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis(self.x,select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num+1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, roi_size=64, reuse=False)

        embs_shape = self.embs.get_shape().as_list()
        with tf.variable_scope("Gaussian_FC") as vs:
            self.app_embs, self.G_var_app_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        if self.sample_app:
            self.embs = self.app_embs
        else:
            # app_embs_one = tf.tile(tf.slice(self.app_embs, [0,0], [1,-1]), [self.batch_size,1])
            app_embs_one = tf.tile(tf.slice(self.embs, [0,0], [1,-1]), [self.batch_size,1])
            self.embs = app_embs_one

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        ## Use py code to get G_pose_inflated, so the op is out of the graph
        # self.G_pose_inflated = tf.placeholder(tf.float32, shape=G_pose.get_shape())
        self.G_pose_inflated = tf_poseInflate(G_pose, self.keypoint_num, radius=4, img_H=self.img_H, img_W=self.img_W)
        with tf.variable_scope("ID_AE") as vs:
            G, _, _ = self.Generator_fn(
                    self.embs_rep, self.G_pose_inflated, 
                    self.channel, self.z_num, self.repeat_num-1, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G = denorm_img(G, self.data_format)

    def test(self):
        test_result_dir = os.path.join(self.model_dir, self.test_dir_name)
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_G_pose = os.path.join(test_result_dir, 'G_pose')
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
        if not os.path.exists(test_result_dir_G_pose):
            os.makedirs(test_result_dir_G_pose)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for i in xrange(self.test_batch_num):
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, mask_fixed, mask_target_fixed, \
                part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if i<4:
                x_fake, G_pose = self.generate(x, pose_fixed, pose_rcv_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=True)
            else:
                x_fake, G_pose = self.generate(x, pose_fixed, pose_rcv_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%04d_c1s1_%06d_%05d.png'%(test_result_dir_G, i, j, idx))
                im = Image.fromarray(p[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                im = Image.fromarray(G_pose[j,:].astype(np.uint8))
                im.save('%s/%04d_%04d.png'%(test_result_dir_G_pose, i, j))
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

    def generate(self, x_fixed, pose_fixed, pose_rcv_fixed, part_bbox_fixed, part_vis_fixed, root_path=None, path=None, idx=None, save=True):
        if self.sample_pose:
            G_pose_rcv, G_pose, reconstruct_loss = self.sess.run([self.G_pose_rcv, self.G_pose, self.reconstruct_loss], \
                                                    {self.pose: pose_fixed, self.part_bbox: part_bbox_fixed, self.part_vis: part_vis_fixed})
            is_normalized=True
        else:
            ## Use reconstructed Pose
            # G_pose_rcv, G_pose, reconstruct_loss = self.sess.run([self.G_pose_rcv, self.G_pose, self.reconstruct_loss], {self.pose: pose_fixed})
            # is_normalized=True
            ## Use random Pose
            G_pose_rcv, G_pose, reconstruct_loss = self.sess.run([self.pose_rcv, self.pose, self.reconstruct_loss])
            G_pose_rcv = np.reshape(G_pose_rcv, [self.batch_size, self.keypoint_num, 3])
            G_pose = np.tile(np.amax((G_pose+1)*127.5, axis=-1, keepdims=True), [1,1,1,3])
            is_normalized=False
            ## Use fixed Pose
            # G_pose_rcv = np.reshape(pose_rcv_fixed, [self.batch_size, self.keypoint_num, 3])
            # G_pose = np.tile(np.amax((pose_rcv_fixed+1)*127.5, axis=-1, keepdims=True), [1,1,1,3])
            # reconstruct_loss = 0.0
            # is_normalized=False
        
        # G_pose_inflated = py_poseInflate(G_pose_rcv, is_normalized=is_normalized, radius=4, img_H=128, img_W=64)
        G, G_pose_inflated = self.sess.run([self.G, self.G_pose_inflated], {self.x: x_fixed, self.part_bbox: part_bbox_fixed})
        G_pose_inflated_img = (np.tile(np.amax(G_pose_inflated, axis=-1, keepdims=True), [1,1,1,3])+1)*127.5
        
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
            path = os.path.join(root_path, '{}_G_pose_reLoss{}.png'.format(idx,reconstruct_loss))
            save_image(G_pose, path)
            print("[*] Samples saved: {}".format(path))
            path = os.path.join(root_path, '{}_G_pose_inflated_reLoss{}.png'.format(idx,reconstruct_loss))
            save_image(G_pose_inflated_img, path)
            print("[*] Samples saved: {}".format(path))
        return G, G_pose_inflated_img