from __future__ import print_function
from trainer import *

class DPIG_FourNets_testOnly(DPIG_Encoder_GAN_BodyROI):    
    def _define_input(self):
        self.train_LuNet = tf.Variable(False, name='phase')
        self.is_train_tensor = tf.Variable(self.is_train, name='phase')
        self.Generator_encoder_fn = models.GeneratorCNN_ID_Encoder
        self.Generator_fn = models.GeneratorCNN_ID_UAEAfterResidual
        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, 
                            LAMBDA=10, G_OUTPUT_DIM=self.img_H*self.img_W*3)
        self.Discriminator_fn = self._getDiscriminator(self.wgan_gp, arch='DCGAN')

    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
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

        # self.test_batch_num = 751
        self.test_batch_num = 24000

        # self.test_dir_name = 'test_result_SampleApp'
        # self.test_dir_name = 'test_result_SamplePose'
        # self.test_dir_name = 'test_result_RandomPose'
        # self.test_dir_name = 'test_result_SampleAppSamplePose'
        self.test_dir_name = 'test_result_SampleAppRandomPose_%dx%d'%(self.test_batch_num, self.batch_size)

    def build_model(self):
        self._define_input()
        # self.pose_rcv = tf.Print(self.pose_rcv, [self.pose_rcv], 'self.pose_rcv = ', summarize=30)
        ################################### Pose ###################################
        with tf.variable_scope("PoseAE") as vs:
            ## Norm to [-1, 1]
            pose_rcv_norm = tf.reshape(self.pose_rcv, [self.batch_size, self.keypoint_num, 3])
            R = tf.cast(tf.slice(pose_rcv_norm, [0,0,0], [-1,-1,1]), tf.float32)/float(self.img_H)*2.0 - 1
            C = tf.cast(tf.slice(pose_rcv_norm, [0,0,1], [-1,-1,1]), tf.float32)/float(self.img_W)*2.0 - 1
            V = tf.cast(tf.slice(pose_rcv_norm, [0,0,2], [-1,-1,1]), tf.float32)
            pose_rcv_norm = tf.concat([R,C,V], axis=-1)
            self.pose_embs, self.G_var_encoder = models.PoseEncoderFCRes(tf.reshape(pose_rcv_norm, [self.batch_size,-1]), 
                        z_num=100, repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU, reuse=False)
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
            self.G_pose_rcv = pose_rcv_norm
            G_pose = coord2channel_simple_rcv(self.G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - self.G_pose_rcv))

        ################################### Appearance ###################################
        ## Use the pose to generate person with pretrained generator
        with tf.variable_scope("Encoder") as vs:
            self.embs, _, self.Encoder_var = GeneratorCNN_ID_Encoder_BodyROI(self.x, self.part_bbox, 7, 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
            embs_shape = self.embs.get_shape().as_list()
        tf.set_random_seed(0)
        with tf.variable_scope("Gaussian_FC") as vs:
            embs_random, self.G_var_app_embs = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)
        if self.one_app_per_batch:
            embs_random = tf.tile(tf.slice(embs_random, [0,0], [1,-1]), [self.batch_size,1])

        if self.sample_app:
            self.embs = embs_random

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
        self.G_dis_score = self.Discriminator_fn(tf.transpose( G, [0,3,1,2] ), input_dim=3)
        ## For PatchDis, we need average
        shape = self.G_dis_score.get_shape().as_list()
        self.G_dis_score = tf.reduce_mean(self.G_dis_score, range(1,len(shape)))

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
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, \
                        mask_fixed, mask_target_fixed, part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if i<4:
                x_fake, G_pose, G_dis_score = self.generate(x, pose_fixed, pose_rcv_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=True)
            else:
                x_fake, G_pose, G_dis_score = self.generate(x, pose_fixed, pose_rcv_fixed, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=False)
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%04d_c1s1_%06d_%05d_%f.png'%(test_result_dir_G, i, j, idx, G_dis_score[j]))
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

    def get_image_from_loader(self):
        x, x_target, pose, pose_target, pose_rcv, pose_rcv_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target = self.sess.run([self.x, self.x_target, self.pose, self.pose_target, 
                        self.pose_rcv, self.pose_rcv_target, self.mask_r6, self.mask_r6_target, self.part_bbox, self.part_bbox_target, self.part_vis, self.part_vis_target])
        x = unprocess_image(x, 127.5, 127.5)
        x_target = unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, pose_rcv, pose_rcv_target, mask, mask_target, part_bbox, part_bbox_target, part_vis, part_vis_target

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
        
        G_pose_inflated = py_poseInflate(G_pose_rcv, is_normalized=is_normalized, radius=4, img_H=128, img_W=64)
        G, G_dis_score = self.sess.run([self.G, self.G_dis_score], {self.x: x_fixed, self.G_pose_inflated: G_pose_inflated, self.part_bbox: part_bbox_fixed})
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
        return G, G_pose_inflated_img, G_dis_score


class DPIG_FourNetsFgBg_testOnly(DPIG_FourNets_testOnly):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            var3 = lib.params_with_name('Discriminator.')
            self.saverPart = tf.train.Saver(var1+var2+var3, max_to_keep=20)

        if self.pretrained_appSample_path is not None:
            var_fg = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Gaussian_FC_Fg')
            var_bg = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Gaussian_FC_Bg')
            self.saverAppSamplePart = tf.train.Saver(var_fg+var_bg, max_to_keep=20)
            
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

        self.test_batch_num = 751
        # self.test_batch_num = 1000
        # self.test_batch_num = 24000

        # self.test_dir_name = 'test_result_SampleApp_FgBg'
        # self.test_dir_name = 'test_result_SamplePose_FgBg'
        # self.test_dir_name = 'test_result_RandomPose_FgBg'
        # self.test_dir_name = 'test_result_SampleAppSamplePose_FgBg'
        # self.test_dir_name = 'test_result_ROI7_PatchDis28x28_RealFgBgRandomPose_FixFgRandBg_pretrain_%dx%d'%(self.test_batch_num, self.batch_size)
        self.test_dir_name = 'test_result_ROI7_RealFgBgRandomPose_FixFgRandBg_pretrain_%dx%d'%(self.test_batch_num, self.batch_size)


    def build_model(self):
        self._define_input()
        # self.pose_rcv = tf.Print(self.pose_rcv, [self.pose_rcv], 'self.pose_rcv = ', summarize=30)
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
            self.G_pose_rcv = pose_rcv_norm
            G_pose = coord2channel_simple_rcv(self.G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - self.G_pose_rcv))

        ################################### Appearance ###################################
        ## Use the pose to generate person with pretrained generator
        self.roi_emb_dim = 32
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 8)
            indices = range(7)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), self.roi_emb_dim, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            self.fg_embs = tf.slice(self.embs, [0,0], [-1,len(indices)*self.roi_emb_dim])
            self.bg_embs = tf.slice(self.embs, [0,len(indices)*self.roi_emb_dim], [-1,-1])

        tf.set_random_seed(0)
        with tf.variable_scope("Gaussian_FC_Fg") as vs:
            embs_shape = self.fg_embs.get_shape().as_list()
            self.app_embs_fg, self.G_var_app_embs_fg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        with tf.variable_scope("Gaussian_FC_Bg") as vs:
            embs_shape = self.bg_embs.get_shape().as_list()
            self.app_embs_bg, self.G_var_app_embs_bg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=256, data_format=self.data_format, activation_fn=LeakyReLU)
        
        if self.one_app_per_batch:
            ## Fix fg and random bg for one person
            fg_one = tf.tile(tf.slice(self.app_embs_fg, [0,0], [1,-1]), [self.batch_size,1])
            embs_random = tf.concat([fg_one, self.app_embs_bg], axis=-1)
            ## Fix fg and bg for one person
            # embs_random = tf.concat([self.app_embs_fg, self.app_embs_bg], axis=-1)
            # embs_random = tf.tile(tf.slice(embs_random, [0,0], [1,-1]), [self.batch_size,1])
        else:
            embs_random = tf.concat([self.app_embs_fg, self.app_embs_bg], axis=-1)

        if self.sample_app:
            self.embs = embs_random
        else:
            if self.one_app_per_batch:
                ## Fix fg and random bg for one person
                fg_one = tf.tile(tf.slice(self.fg_embs, [0,0], [1,-1]), [self.batch_size,1])
                embs_random = tf.concat([fg_one, self.bg_embs], axis=-1)
                self.embs = embs_random

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

        self.G_dis_score = self.Discriminator_fn(tf.transpose( G, [0,3,1,2] ), input_dim=3)
        ## For PatchDis, we need average
        shape = self.G_dis_score.get_shape().as_list()
        self.G_dis_score = tf.reduce_mean(self.G_dis_score, range(1,len(shape)))



class DPIG_FourNetsFgBg_testOnlySampleFactor(DPIG_FourNets_testOnly):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            var3 = lib.params_with_name('Discriminator.')
            self.saverPart = tf.train.Saver(var1+var2+var3, max_to_keep=20)

        if self.pretrained_appSample_path is not None:
            var_fg = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Gaussian_FC_Fg')
            var_bg = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Gaussian_FC_Bg')
            self.saverAppSamplePart = tf.train.Saver(var_fg+var_bg, max_to_keep=20)
            
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

        # self.test_batch_num = 20  ## For test
        self.test_batch_num = 400 ## For IS score
        self.test_dir_name = 'test_result_ROI7_SampleFg%rSampleBg%rSamplePose%r_pretrain_%dx%d'%(self.sample_fg,self.sample_bg,self.sample_pose, self.test_batch_num, self.batch_size)
        # self.test_dir_name = 'test_result_ROI7_SampleFg%rSampleBg%rRealPose%r_pretrain_%dx%d'%(self.sample_fg,self.sample_bg,self.sample_pose, self.test_batch_num, self.batch_size)

    def build_model(self):
        self._define_input()
        # self.pose_rcv = tf.Print(self.pose_rcv, [self.pose_rcv], 'self.pose_rcv = ', summarize=30)
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
            # self.G_pose_rcv = pose_rcv_norm
            G_pose = coord2channel_simple_rcv(self.G_pose_rcv, self.keypoint_num, is_normalized=True, img_H=self.img_H, img_W=self.img_W)
            self.G_pose = denorm_img(tf.tile(tf.reduce_max(G_pose, axis=-1, keep_dims=True), [1,1,1,3]), self.data_format)

        self.reconstruct_loss = tf.reduce_mean(tf.square(pose_rcv_norm - self.G_pose_rcv))

        ################################### Appearance ###################################
        ## Use the pose to generate person with pretrained generator
        self.roi_emb_dim = 32
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 8)
            indices = range(7)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), self.roi_emb_dim, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)
            self.fg_embs = tf.slice(self.embs, [0,0], [-1,len(indices)*self.roi_emb_dim])
            self.bg_embs = tf.slice(self.embs, [0,len(indices)*self.roi_emb_dim], [-1,-1])

        ## FG
        tf.set_random_seed(0)
        with tf.variable_scope("Gaussian_FC_Fg") as vs:
            embs_shape = self.fg_embs.get_shape().as_list()
            self.app_embs_fg, self.G_var_app_embs_fg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=512, data_format=self.data_format, activation_fn=LeakyReLU)

        if self.sample_fg:
            self.embs_fg = self.app_embs_fg
        else:
            fg_one = tf.tile(tf.slice(self.fg_embs, [0,0], [1,-1]), [self.batch_size,1])
            self.embs_fg = fg_one

        ## BG
        with tf.variable_scope("Gaussian_FC_Bg") as vs:
            embs_shape = self.bg_embs.get_shape().as_list()
            self.app_embs_bg, self.G_var_app_embs_bg = models.GaussianFCRes(embs_shape, embs_shape[-1], repeat_num=4, hidden_num=256, data_format=self.data_format, activation_fn=LeakyReLU)
        
        if self.sample_bg:
            self.embs_bg = self.app_embs_bg
        else:
            bg_one = tf.tile(tf.slice(self.bg_embs, [0,0], [1,-1]), [self.batch_size,1])
            self.embs_bg = bg_one

        self.embs = tf.concat([self.embs_fg, self.embs_bg], axis=-1)
        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        ## Use py code to get G_pose_inflated, so the op is out of the graph
        # self.G_pose_inflated = tf.placeholder(tf.float32, shape=G_pose.get_shape())
        self.G_pose_inflated = tf_poseInflate(G_pose, self.keypoint_num, radius=4, img_H=128, img_W=64)
        with tf.variable_scope("ID_AE") as vs:
            G, _, _ = self.Generator_fn(
                    self.embs_rep, self.G_pose_inflated, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G = denorm_img(G, self.data_format)

        self.G_dis_score = self.Discriminator_fn(tf.transpose( G, [0,3,1,2] ), input_dim=3)
        ## For PatchDis, we need average
        shape = self.G_dis_score.get_shape().as_list()
        self.G_dis_score = tf.reduce_mean(self.G_dis_score, range(1,len(shape)))

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
        G, G_dis_score, G_pose_inflated = self.sess.run([self.G, self.G_dis_score, self.G_pose_inflated], {self.x: x_fixed, self.part_bbox: part_bbox_fixed})
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
        return G, G_pose_inflated_img, G_dis_score


class DPIG_FourNetsFgBg_testOnlyCondition(DPIG_FourNets_testOnly):
    def init_net(self):
        self.build_model()

        if self.pretrained_path is not None:
            var1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Encoder')
            var2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ID_AE')
            var3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Discriminator.')
            self.saverPart = tf.train.Saver(var1+var2+var3, max_to_keep=20)

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

        # self.test_batch_num = 10 ## for test_samples
        self.test_batch_num = 600 ## for test

        if self.sample_pose:
            self.test_dir_name = 'test_samples_result_ROI7_Condition_TargetPose_%dx%d'%(self.test_batch_num, self.batch_size)
        else:
            self.test_dir_name = 'test_samples_result_ROI7_Condition_TargetPose_%dx%d'%(self.test_batch_num, self.batch_size)

    def build_model(self):
        self._define_input()
        ################################### Appearance ###################################
        ## Use the pose to generate person with pretrained generator
        with tf.variable_scope("Encoder") as vs:
            pb_list = tf.split(self.part_bbox, self.part_num, axis=1)
            pv_list = tf.split(self.part_vis, self.part_num, axis=1)
            ## Part 1-7 (totally 8)
            indices = range(7)
            ## Part 1,8-16 (totally 10)
            # indices = [0] + range(7,16)
            select_part_bbox = tf.concat([pb_list[i] for i in indices], axis=1)
            select_part_vis = tf.cast(tf.concat([pv_list[i] for i in indices], axis=1), tf.float32)
            self.embs, _, _, self.Encoder_var = models.GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(self.x, self.mask_r6, select_part_bbox, select_part_vis, len(indices), 32, 
                                            self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, keep_part_prob=1.0, reuse=False)

        self.embs_rep = tf.tile(tf.expand_dims(self.embs,-1), [1, 1, self.img_H*self.img_W])
        self.embs_rep = tf.reshape(self.embs_rep, [self.batch_size, -1, self.img_H, self.img_W])
        self.embs_rep = nchw_to_nhwc(self.embs_rep)

        with tf.variable_scope("ID_AE") as vs:
            G, _, _ = self.Generator_fn(
                    self.embs_rep, self.pose_target, 
                    self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, reuse=False)
        self.G = denorm_img(G, self.data_format)

        self.G_dis_score = self.Discriminator_fn(tf.transpose( G, [0,3,1,2] ), input_dim=3)
        ## For PatchDis, we need average
        shape = self.G_dis_score.get_shape().as_list()
        self.G_dis_score = tf.reduce_mean(self.G_dis_score, range(1,len(shape)))

    def generate(self, x_fixed, x_target_fixed, pose_target_fixed, mask_fixed, part_bbox_fixed, part_vis_fixed, root_path=None, path=None, idx=None, save=True):
        G, G_dis_score = self.sess.run([self.G, self.G_dis_score], {self.x: x_fixed, self.pose_target: pose_target_fixed, \
                            self.mask_r6: mask_fixed, self.part_bbox: part_bbox_fixed, self.part_vis: part_vis_fixed})
        
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
        return G, G_dis_score

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
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, pose_rcv_fixed, pose_rcv_target_fixed, \
                        mask_fixed, mask_target_fixed, part_bbox_fixed, part_bbox_target_fixed, part_vis_fixed, part_vis_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            mask = mask_fixed/255
            if 0==i:
                x_fake, G_dis_score = self.generate(x, x_target, pose_target_fixed, mask, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=True)
            else:
                x_fake, G_dis_score = self.generate(x, x_target, pose_target_fixed, mask, part_bbox_fixed, part_vis_fixed, test_result_dir, idx=i, save=False)
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
                # im.save('%s/%04d_c1s1_%06d_%05d_%f.png'%(test_result_dir_G, i, j, idx, G_dis_score[j]))
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




################################################################################################
####################################### DF test models #########################################

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