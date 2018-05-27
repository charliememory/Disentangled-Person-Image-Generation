source ~/.bashrc

if [ ! -d ./data/DF_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip
    unzip DF_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_sparse_partBbox37_maskR4R8_roi10Complete DF_train_data
    rm -f DF_train_data.zip
    cd ..
fi
if [ ! -d ./data/DF_trainAStest_data ]; then
    cd data
    mkdir DF_trainAStest_data
    cd DF_trainAStest_data
    ln -s ../DF_train_data/* .
    for file in *train* ; do mv "$file" "${file/train/test}" ; done
    cd ../..
fi

if [ ! -d ./data/DF_test_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip
    unzip DF_test_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_test_sparse_partBbox37_maskR4R8_roi10Complete DF_test_data
    rm -f DF_test_data.zip
    cd ..
fi

gpu=0
D_arch='DCGAN'
log_dir=your_log_dir_path
log_dir_pretrain=your_pretrained_log_dir_path

####################### Testing Whole Framework #####################
model_dir=${log_dir}'/MODEL104_DF_subnetSamplePoseRCV_WGAN'
## Appearance
pretrained_path=${log_dir_pretrain}'/MODEL101_DF_Encoder_GAN_BodyROI7_App/model.ckpt-0'
pretrained_appSample_path=${log_dir_pretrain}'/MODEL103_DF_subnetSampleApp_WGAN/model.ckpt-0'
pretrained_poseAE_path=${log_dir_pretrain}'/MODEL102_DF_PoseRCV_AE/model.ckpt-0'
pretrained_poseSample_path=${log_dir_pretrain}'/MODEL104_DF_subnetSamplePoseRCV_WGAN/model.ckpt-0'

## Generate data for Sampling one or more factors
python main.py --dataset=DF_test_data \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=16 \
             --is_train=False \
             --sample_app=False \
             --sample_pose=True \
             --model=1002 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_path=${pretrained_path} \
             --pretrained_appSample_path=${pretrained_appSample_path} \
             --pretrained_poseAE_path=${pretrained_poseAE_path} \
             --pretrained_poseSample_path=${pretrained_poseSample_path} \

## PG2 task (Conditional pose guided person image generation)
model_dir=${log_dir}'/MODEL101_DF_Encoder_GAN_BodyROI7_App'
pretrained_path=${log_dir_pretrain}'/MODEL101_DF_Encoder_GAN_BodyROI7_App/model.ckpt-0'

python main.py --dataset=DF_test_data \
             --use_gpu=True --img_H=256  --img_W=256 \
             --batch_size=16 \
             --is_train=False \
             --model=1001 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_path=${pretrained_path} \

## Score
test_dir=your_test_dir_name
stage=1
python score.py ${stage} ${gpu} ${model_dir} ${test_dir}
python score_mask.py ${stage} ${gpu} ${model_dir} ${test_dir}