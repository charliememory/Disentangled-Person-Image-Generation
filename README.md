# Disentangled-Person-Image-Generation
Tensorflow implementation of CVPR 2018 paper [Disentangled Person Image Generation](http://homes.esat.kuleuven.be/~liqianma/pdf/CVPR18_Ma_Disentangled_Person_Image_Generation.pdf)

## Two stages framework
<p align="center">
<img src="https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_adver_comb.svg" width="600"/></p>

### Stage-I: reconstruction
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_recons_large.svg)

### Stage-II: sampling
![](https://github.com/charliememory/Disentangled-Person-Image-Generation/blob/master/imgs/Paper_Framework_sampling.svg)

## Resources
 - Pretrained models: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/models/Market1501_models.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/models/DF_models.zip).
 - Training data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip).
 - Testing data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_test_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip).
 - Raw data: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market1501_img_pose_attr_seg.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_img_pose.zip) 


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