source ~/.bashrc

if [ ! -d ./data/Market_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip
    unzip Market_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_128x64PoseRCV_Mask_sparse_Attr_partBbox7_maskR4R6 Market_train_data
    rm -f Market_train_data.zip
    cd ..
fi

gpu=0
D_arch='DCGAN'
log_dir=your_log_dir_path
log_dir_pretrain=your_pretrained_log_dir_path

####################### Stage-I: reconstruction #####################
## Fg Bg reconstruction
model_dir=${log_dir}'/MODEL1_Encoder_GAN_BodyROI7'
python main.py --dataset=Market_train_data \
             --use_gpu=True --img_H=128  --img_W=64 \
             --batch_size=16 --max_step=120000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=1 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \

# PoseRCV reconstruction
model_dir=${log_dir}'/MODEL2_PoseRCV_AE'
python main.py --dataset=Market_train_data \
             --use_gpu=True --img_H=128  --img_W=64 \
             --batch_size=64 --max_step=60000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=2 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \


####################### Stage-II: sampling #####################
## FgBg sampling
model_dir=${log_dir}'/MODEL3_subSampleAppNetFgBg_WGAN'
pretrained_path=${log_dir_pretrain}'/MODEL1_Encoder_GAN_BodyROI7_PartVis_FgBg/model.ckpt-xxx'
python main.py --dataset=Market_train_data \
             --use_gpu=True --img_H=128  --img_W=64 \
             --batch_size=32 --max_step=120000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=3 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=0  --pretrained_path=${pretrained_path} \


## Pose sampling
model_dir=${log_dir}'/MODEL4_subnetSamplePoseRCV_WGAN'
pretrained_poseSample_path=${log_dir_pretrain}'/MODEL2_PoseRCV_AE/model.ckpt-xxx'
python main.py --dataset=Market_train_data \
             --use_gpu=True --img_H=128  --img_W=64 \
             --batch_size=64 --max_step=60000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --model=4 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=0  --pretrained_path=${pretrained_path} \