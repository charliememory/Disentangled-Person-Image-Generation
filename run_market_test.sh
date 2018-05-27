source ~/.bashrc

if [ ! -d ./data/Market_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip
    unzip Market_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_128x64PoseRCV_Mask_sparse_Attr_partBbox7_maskR4R6 Market_train_data
    rm -f Market_train_data.zip
    cd ..
fi
if [ ! -d ./data/Market_trainAStest_data ]; then
    cd data
    mkdir Market_trainAStest_data
    cd Market_trainAStest_data
    ln -s ../Market_train_data/* .
    for file in *train* ; do mv "$file" "${file/train/test}" ; done
    cd ../..
fi

if [ ! -d ./data/Market_test_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_test_data.zip
    unzip Market_test_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_128x64PoseRCV_Mask_test_sparse_Attr_partBbox7_maskR4R6 Market_test_data
    rm -f Market_test_data.zip
    cd ..
fi

gpu=0
D_arch='DCGAN'
log_dir='./logs'
log_dir_pretrain='/esat/diamond/liqianma/HomepageResources/CVPR18_DPIG/models_original'
stage=1


####################### Testing Whole Framework #####################
model_dir=${log_dir}'/MODEL4_subnetSamplePoseRCV_WGAN'
## Appearance
pretrained_path=${log_dir_pretrain}'/MODEL1_Encoder_GAN_BodyROI7_PartVis_FgBg/model.ckpt-0'
pretrained_appSample_path=${log_dir_pretrain}'/MODEL3_subSampleAppNetFgBg_WGAN/model.ckpt-0'
## Pose
pretrained_poseAE_path=${log_dir_pretrain}'/MODEL2_PoseRCV_AE/model.ckpt-0'
pretrained_poseSample_path=${log_dir_pretrain}'/MODEL4_subnetSamplePoseRCV_WGAN/model.ckpt-0'

## Generate data for re-id
python main.py --dataset=Market_trainAStest_data \
             --use_gpu=True --input_scale_size=128 \
             --batch_size=32 \
             --is_train=False \
             --sample_app=True \
             --sample_pose=False \
             --one_app_per_batch=True \
             --model=11 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_path=${pretrained_path} \
             --pretrained_appSample_path=${pretrained_appSample_path} \
             --pretrained_poseAE_path=${pretrained_poseAE_path} \
             --pretrained_poseSample_path=${pretrained_poseSample_path} \


## Generate data for Sampling one or more factors
python main.py --dataset=Market_trainAStest_data \
             --use_gpu=True --input_scale_size=128 \
             --batch_size=32 \
             --is_train=False \
             --sample_fg=True \
             --sample_bg=True \
             --sample_pose=True \
             --model=13 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_path=${pretrained_path} \
             --pretrained_appSample_path=${pretrained_appSample_path} \
             --pretrained_poseAE_path=${pretrained_poseAE_path} \
             --pretrained_poseSample_path=${pretrained_poseSample_path} \

## PG2 task (Conditional pose guided person image generation)
python main.py --dataset=Market_test_data \
             --use_gpu=True --input_scale_size=128 \
             --batch_size=32 \
             --is_train=False \
             --model=12 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --pretrained_path=${pretrained_path} \
             --pretrained_poseAE_path=${pretrained_poseAE_path} \
             --pretrained_poseSample_path=${pretrained_poseSample_path} \

test_dir=your_test_dir_name
python score.py ${stage} ${gpu} ${model_dir} ${test_dir}
python score_mask.py ${stage} ${gpu} ${model_dir} ${test_dir}

