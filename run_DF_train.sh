source ~/.bashrc

if [ ! -d ./data/DF_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip
    unzip DF_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_sparse_partBbox37_maskR4R8_roi10Complete DF_train_data
    rm -f DF_train_data.zip
    cd ..
fi

gpu=0
D_arch='DCGAN'
log_dir=your_log_dir_path
log_dir_pretrain=your_pretrained_log_dir_path


####################### Stage-I: reconstruction #####################
## Appearance reconstruction
model_dir=${log_dir}'/MODEL101_DF_Encoder_GAN_BodyROI7'
python main.py --dataset=DF_train_data \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=6 --max_step=120000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=101 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \

## PoseRCV reconstruction
model_dir=${log_dir}'/MODEL102_DF_PoseRCV_AE'
python main.py --dataset=DF_train_data \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=16 --max_step=120000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=102 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \


####################### Stage-II: sampling #####################
## Appearance sampling
model_dir=${log_dir}'/MODEL103_DF_subSampleAppNet_WGAN'
pretrained_path=${log_dir_pretrain}'/MODEL101_DF_Encoder_GAN_BodyROI7/model.ckpt-0'
python main.py --dataset=DF_train_data \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=16 --max_step=120000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=103 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=0  --pretrained_path=${pretrained_path} \

## Pose sampling
model_dir=${log_dir}'/MODEL104_DF_subnetSamplePoseRCV_WGAN'
pretrained_path=${log_dir_pretrain}'/MODEL101_DF_Encoder_GAN_BodyROI7/model.ckpt-0'
pretrained_poseAE_path=${log_dir_pretrain}'/MODEL102_DF_PoseRCV_AE/model.ckpt-0' 
python main.py --dataset=deepfashion_train_pose_onlyPosPair_256x256PoseRCV_Mask_sparse_partBbox37_maskR4R8_roi10Complete \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=32 --max_step=60000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=104 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_poseAE_path=${pretrained_poseAE_path} \
             --start_step=0  --pretrained_path=${pretrained_path} \