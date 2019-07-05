import scipy.io
import numpy as np
import os, sys, pdb, pickle

######## Mask-RCNN keypoint order ########
# % 1: nose
# % 2: left eye
# % 3: right eye
# % 4: left ear
# % 5: right ear
# % 6: left shoulder
# % 7: right shoulder
# % 8: left elbow
# % 9: right elbow
# % 10: left wrist
# % 11: right wrist
# % 12: left hip
# % 13: right hip
# % 14: left knee
# % 15: right knee
# % 16: left ankle
# % 17: right ankle

######## OpenPose  keypoint order ########
# MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]

keyNum = 18
openPose_maskRCNN_trans_dic = {0:0, 1:None, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:1, 15:2, 16:3, 17:4}
def mat2dic(img_dir, pose_mat_path):
    pose_mat = scipy.io.loadmat(pose_mat_path)['joint2d']
    N, _, _ = pose_mat.shape
    img_name_list = sorted(os.listdir(img_dir))
    assert N==len(img_name_list), 'number of pose and img are different'

    pose_dic = {}
    for idx, img_name in enumerate(img_name_list):
        crs = pose_mat[idx,:,:]
        RCV = np.zeros([keyNum, 3])
        for k in range(keyNum):
            k_idx = openPose_maskRCNN_trans_dic[k]
            if k_idx is not None:
                c,r = crs[:,k_idx]
                if not (0==c and 0==r):
                    RCV[k,0] = r
                    RCV[k,1] = c
                    RCV[k,2] = 1  ## 1 means visible, 0 means invisible
            ## Makeup neck keypoint with leftShoulder and rightShoulder
            r0, c0, v0 = RCV[2,:]
            r1, c1, v1 = RCV[5,:]
            if v0 and v1:
                RCV[1,0] = (r0+r1)/2
                RCV[1,1] = (c0+c1)/2
                RCV[1,2] = 1
        pose_dic[img_name] = RCV

    save_path = os.path.join(os.path.dirname(pose_mat_path), os.path.basename(pose_mat_path).split('_')[-1].replace('.mat','.pickle'))
    with open(save_path, 'w') as f:
        pickle.dump(pose_dic, f)

img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/test/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/PoseRCV/marc_test.mat'
mat2dic(img_dir, pose_mat_path)
img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/train/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/marc/PoseRCV/marc_train.mat'
mat2dic(img_dir, pose_mat_path)
img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/test/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/PoseRCV/weipeng_test.mat'
mat2dic(img_dir, pose_mat_path)
img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/train/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/white_bg/weipeng/PoseRCV/weipeng_train.mat'
mat2dic(img_dir, pose_mat_path)
