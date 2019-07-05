# Disentangled-Person-Image-Generation
Tensorflow implementation of CVPR 2018 paper [Disentangled Person Image Generation](http://homes.esat.kuleuven.be/~liqianma/pdf/CVPR18_Ma_Disentangled_Person_Image_Generation.pdf)

## Two stages framework
<p align="center">
<img src="https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_adver_comb.svg" width="600"/></p>

### Stage-I: reconstruction
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_recons_large.svg)

## Sampling phase
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_sampling.svg)

## Dependencies
- python 2.7
- tensorflow-gpu (1.4.1)
- numpy (1.14.0)
- Pillow (5.0.0)
- scikit-image (0.13.0)
- scipy (1.0.1)
- matplotlib (2.0.0)

## Resources
 - Pretrained models: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/models/Market1501_models.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/models/DF_models.zip).
 - Training data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip).
 - Testing data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_test_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip).
 - Raw data: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market1501_img_pose_attr_seg.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_img_pose.zip) 
 - Virtual Market Dataset with 500 ID x 24 images: [VirtualMarket](https://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/data/VirtualMarket_500x24.zip)
<p align="center">
<img src="https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_virtual_market_pair.svg" width="600"/></p>

## TF-record data preparation steps
 You can skip this data preparation procedure if directly using the tf-record data files.
 1. `cd datasets`
 2. `./run_convert_market.sh` to download and convert the original images, poses, attributes, segmentations
 3. `./run_convert_DF.sh` to download and convert the original images, poses
 4. \[Optional\] `./run_convert_RCV.sh` to convert the original images and pose coordinates, i.e. `(row, column, visibility)` (e.g. from OpenPose or MaskRCNN), which can be useful for other datasets.
 Note: we also provide the convert code for [Market-1501 Attribute](https://github.com/vana77/Market-1501_Attribute) and Market-1501 Segmentation results from [PSPNet](https://github.com/hszhao/PSPNet). These extra info. are provided for further research. In our experiments, pose mask are ontained from pose key-points (see `_getPoseMask` function in convert .py files).

## Training steps
 1. Download the tf-record training data.
 2. Modify the `log_dir` and `log_dir_pretrain` in the run_market_train.sh/run_DF_train.sh scripts.
 3. run run_market_train.sh/run_DF_train.sh 
 
 Note: we use a triplet instead of pair real/fake for adversarial training to keep training more stable.

## Testing steps
 1. Download the pretrained models and tf-record testing data.
 2. Modify the `log_dir` and `log_dir_pretrain` in the run_market_test.sh/run_DF_test.sh scripts.
 3. run run_market_test.sh/run_DF_test.sh 


## Fg/Bg/Pose sampling on Market-1501
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Sampling_market.svg)

## Appearance sampling on DeepFashion dataset
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/supp_DF_sampling_app.svg)

## Pose sampling on DeepFashion dataset
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/supp_DF_sampling_pose.svg)

## Pose interpolation between real images
 - Between same person:
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_inverse_interpolate.svg)
 - Between different persons:
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/supp_inverse_interpolate_diff_person.svg)

## Pose guided person image generation
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_comparison_PG2.svg)

## Citation
```
@inproceedings{ma2018disentangled,
  title={Disentangled Person Image Generation},
  author={Ma, Liqian and Sun, Qianru and Georgoulis, Stamatios and Van Gool, Luc and Schiele, Bernt and Fritz, Mario},
  booktitle={{IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Related projects
- [Pose-Guided-Person-Image-Generation](https://github.com/charliememory/Pose-Guided-Person-Image-Generation)
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [improved_wgan_training](https://github.com/igul222/improved_wgan_training)
