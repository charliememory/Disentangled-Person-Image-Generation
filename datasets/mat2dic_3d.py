######## Mask-RCNN keypoint order ########
# left_knee     
# left_ankle    
# right_knee    
# right_ankle   
# left_shoulder 
# left_elbow    
# left_hand     
# right_shoulder
# right_elbow   
# right_hand    
# neck          
# head          
# left_hip      
# right_hip     
# spine         
# right_foot    
# left_foot     
# lefteye      
# righteye     
# nose         
# chin         
# leftear
# rightear       

######## OpenPose  keypoint order ########
# MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
import scipy.io
import numpy as np
import os, sys, pdb, pickle

keyNum = 18
openPose_maskRCNN_trans_dic = {0:19, 1:10, 2:7, 3:8, 4:9, 5:4, 6:5, 7:6, 8:13, 9:2, 10:3, 11:12, 12:0, 13:1, 14:17, 15:18, 16:21, 17:22}
def mat2dic(img_dir, pose_mat_path):
    pose_mat = scipy.io.loadmat(pose_mat_path)[os.path.basename(pose_mat_path).split('.')[0]]
    pose_mat = pose_mat.reshape([-1,23,2]).transpose([0,2,1])
    # pdb.set_trace()
    N, _, _ = pose_mat.shape
    img_name_list = sorted(os.listdir(img_dir))
    # assert N==len(img_name_list), 'number of pose and img are different'

    pose_dic = {}
    # for idx, img_name in enumerate(img_name_list):
    for idx in range(N):
        img_name = img_name_list[idx]
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

    # save_path = os.path.join(os.path.dirname(pose_mat_path), 'test_rescaled.pickle')
    save_path = os.path.join(os.path.dirname(pose_mat_path), 'test.pickle')
    with open(save_path, 'w') as f:
        pickle.dump(pose_dic, f)

img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc/test/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/marc/PoseRCV/marc_drive_weipeng.mat'
mat2dic(img_dir, pose_mat_path)
img_dir = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng/test/'
pose_mat_path = '/esat/diamond/liqianma/datasets/Pose/MPI_CG/complex_bg/weipeng/PoseRCV/weipeng_drive_marc.mat'
mat2dic(img_dir, pose_mat_path)