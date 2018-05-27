import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import pdb
# import utils_wgan
from utils import *

import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
from tensorflow.python.framework import ops
    
#################### MS_SSIM Loss #####################
## ref code: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
###########################################################


#################### Bernoulli Sample #####################
## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
@tf.RegisterGradient("ConstGrad")
def _const_grad(op, grad):
    return grad

def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        # with g.gradient_override_map({"Round": "ConstGrad"}):
        #     return tf.round(x, name=name)
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)
        # For Tensorflow v0.11 and below use:
        #with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)

def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (identity).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]



def softMargin(x):
    return tf.log1p(tf.exp(x))

def LeakyReLU(x, alpha=0.3):
    return tf.maximum(alpha*x, x)


def Batchnorm(inputs, is_training, name=None, data_format='NHWC'):
    bn = tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=name, 
                                      data_format=data_format)
    return bn

def Layernorm(name, axes, inputs):
    # pdb.set_trace()
    return lib.ops.layernorm.Layernorm(name,axes,inputs)

## Ref code: https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
def Instance_norm(net, data_format='NHWC'):
    if 'NHWC'==data_format:
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    elif 'NCHW'==data_format:
        batch, channels, rows, cols = [i.value for i in net.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [2,3], keep_dims=True)
    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

## Ref code: https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py
def ResBottleNeckBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 1, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

def ResBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 3, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

## Ref code: https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py
def ResBottleNeckBlockBN(x, n1, n2, n3, data_format, is_train_tensor, activation_fn=LeakyReLU):
    preact = activation_fn(Batchnorm(x, is_train_tensor))
    if n1 != n3:
        shortcut = slim.conv2d(preact, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    residual = slim.conv2d(preact, n2, 1, 1, activation_fn=activation_fn, data_format=data_format)
    residual = slim.conv2d(residual, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    residual = slim.conv2d(residual, n3, 1, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + residual)
    return out

def ResBlockBN(x, n1, n2, n3, data_format, is_train_tensor, activation_fn=LeakyReLU):
    preact = activation_fn(Batchnorm(x, is_train_tensor))
    # preact = x
    if n1 != n3:
        shortcut = slim.conv2d(preact, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    residual = slim.conv2d(preact, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    # residual = activation_fn(Batchnorm(residual, is_train_tensor))
    residual = slim.conv2d(residual, n3, 3, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + residual)
    return out

def ConvBnLeakyReLU(x, out_channel, kernel_size, stride, data_format, is_train_tensor, alpha=0.2):
    x = slim.conv2d(x, out_channel, kernel_size, stride, activation_fn=None, data_format=data_format)
    x = Batchnorm(x, is_train_tensor)
    out = LeakyReLU(x, alpha)
    return out


#################################### Generator ####################################
def GeneratorCNN_ID_Encoder(x, pose, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, reuse=False):
    with tf.variable_scope("G_encoder",reuse=reuse) as vs:
        if pose is not None:
            if data_format == 'NCHW':
                x = tf.concat([x, pose], 1)
            elif data_format == 'NHWC':
                x = tf.concat([x, pose], 3)

        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
        out = slim.fully_connected(x, z_num, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def GeneratorCNN_ID_Decoder(x, out_H, out_W, out_channel, repeat_num, hidden_num, data_format, activation_fn=tf.nn.relu, reuse=False):
    with tf.variable_scope("G_Decoder",reuse=reuse) as vs:
        # Decoder
        in_H = out_H/(2**(repeat_num-1))
        in_W = out_W/(2**(repeat_num-1))
        x = slim.fully_connected(x, np.prod([in_H, in_W, hidden_num*repeat_num]), activation_fn=activation_fn)
        x = reshape(x, in_H, in_W, hidden_num*repeat_num, data_format)
        # pdb.set_trace()
        
        for idx in range(repeat_num):
            res = x
            channel_num = hidden_num * (repeat_num-idx)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = res + x
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn, data_format=data_format)
        out = slim.conv2d(x, out_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def GeneratorCNN_ID_Encoder_BodyROI(x, ROI_bboxs, bbox_num, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, keep_part_prob=1.0, roi_size=48, reuse=False):
    with tf.variable_scope("G_encoder",reuse=reuse) as vs:
        batch_num = x.get_shape().as_list()[0]
        img_H = float(x.get_shape().as_list()[1])
        img_W = float(x.get_shape().as_list()[2])
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        res = x
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = x + res

        body_roi_list = []
        for i in range(bbox_num):
            bbox = tf.cast(tf.slice(ROI_bboxs, [0,i,0], [-1,1,-1]), tf.float32) ## [y1,x1,y2,x2]
            bbox = tf.squeeze(bbox, axis=1)
            y1 = tf.slice(bbox, [0,0], [-1,1])/img_H
            x1 = tf.slice(bbox, [0,1], [-1,1])/img_W
            y2 = tf.slice(bbox, [0,2], [-1,1])/img_H
            x2 = tf.slice(bbox, [0,3], [-1,1])/img_W
            bbox = tf.concat([y1,x1,y2,x2], axis=-1)
            body_roi_list.append(tf.image.crop_and_resize(x, bbox, range(batch_num), [roi_size, roi_size]))

        ## Share weights for different body regions
        body_regions = tf.concat(body_roi_list, axis=0)
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = body_regions
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = body_regions + res
            if idx < repeat_num - 1:
                body_regions = slim.conv2d(body_regions, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
        body_regions = tf.reshape(body_regions, [body_regions.get_shape().as_list()[0], -1])
        body_regions = slim.fully_connected(body_regions, z_num, activation_fn=None)

        fea_list = tf.split(body_regions, bbox_num, axis=0)
        if keep_part_prob<1.0:
            ## Randomly drop the part feature
            for i in range(bbox_num):
                mask = bernoulliSample([keep_part_prob])
                mask = tf.expand_dims(mask, axis=0)
                # mask = tf.Print(mask, [mask], 'mask = ', summarize=10)
                mask = tf.tile(mask, [batch_num, z_num])
                fea_list[i] *= mask
        fea_all = tf.concat(fea_list, axis=-1)

    variables = tf.contrib.framework.get_variables(vs)
    return fea_all, fea_list, variables


def GeneratorCNN_ID_Encoder_BodyROIVis(x, ROI_bboxs, ROI_vis, bbox_num, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, keep_part_prob=1.0, roi_size=48, reuse=False):
    with tf.variable_scope("G_encoder",reuse=reuse) as vs:
        batch_num = x.get_shape().as_list()[0]
        img_H = float(x.get_shape().as_list()[1])
        img_W = float(x.get_shape().as_list()[2])
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        res = x
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = x + res

        body_roi_list = []
        for i in range(bbox_num):
            bbox = tf.cast(tf.slice(ROI_bboxs, [0,i,0], [-1,1,-1]), tf.float32) ## [y1,x1,y2,x2]
            bbox = tf.squeeze(bbox, axis=1)
            y1 = tf.slice(bbox, [0,0], [-1,1])/img_H
            x1 = tf.slice(bbox, [0,1], [-1,1])/img_W
            y2 = tf.slice(bbox, [0,2], [-1,1])/img_H
            x2 = tf.slice(bbox, [0,3], [-1,1])/img_W
            bbox = tf.concat([y1,x1,y2,x2], axis=-1)
            body_roi_list.append(tf.image.crop_and_resize(x, bbox, range(batch_num), [roi_size, roi_size]))

        ## Share weights for different body regions
        body_regions = tf.concat(body_roi_list, axis=0)
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = body_regions
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = body_regions + res
            if idx < repeat_num - 1:
                body_regions = slim.conv2d(body_regions, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
        body_regions = tf.reshape(body_regions, [body_regions.get_shape().as_list()[0], -1])
        body_regions = slim.fully_connected(body_regions, z_num, activation_fn=None)

        fea_list = tf.split(body_regions, bbox_num, axis=0)
        ## ROI_vis
        ROI_vis_list = tf.split(ROI_vis, bbox_num, axis=1)
        for i in range(bbox_num):
            mask = ROI_vis_list[i]
            # mask = tf.expand_dims(mask, axis=0)
            # pdb.set_trace()
            # mask = tf.Print(mask, [mask], 'ROI_vis mask = ', summarize=10)
            mask = tf.tile(mask, [1, z_num])
            fea_list[i] *= mask
        ## Randomly drop part
        if keep_part_prob<1.0:
            ## Randomly drop the part feature
            for i in range(bbox_num):
                mask = bernoulliSample([keep_part_prob]*batch_num)
                mask = tf.expand_dims(mask, axis=1)
                # mask = tf.Print(mask, [mask], 'keep_part_prob mask = ', summarize=10)
                mask = tf.tile(mask, [1, z_num])
                fea_list[i] *= mask
        fea_all = tf.concat(fea_list, axis=-1)

    variables = tf.contrib.framework.get_variables(vs)
    return fea_all, fea_list, variables

def GeneratorCNN_ID_Encoder_BodyROIVis_FgBgFeaTwoBranch(x, fg_mask, ROI_bboxs, ROI_vis, bbox_num, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, keep_part_prob=1.0, roi_size=48, reuse=False):
    with tf.variable_scope("G_encoder",reuse=reuse) as vs:
        batch_num = x.get_shape().as_list()[0]
        img_H = float(x.get_shape().as_list()[1])
        img_W = float(x.get_shape().as_list()[2])
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        res = x
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        x = x + res

        x_fg = x * tf.cast(fg_mask, tf.float32)
        x_bg = x * (1.0-tf.cast(fg_mask, tf.float32))

        body_roi_list = []
        for i in range(bbox_num):
            bbox = tf.cast(tf.slice(ROI_bboxs, [0,i,0], [-1,1,-1]), tf.float32) ## [y1,x1,y2,x2]
            # pdb.set_trace()
            bbox = tf.squeeze(bbox, axis=1)
            y1 = tf.slice(bbox, [0,0], [-1,1])/img_H
            x1 = tf.slice(bbox, [0,1], [-1,1])/img_W
            y2 = tf.slice(bbox, [0,2], [-1,1])/img_H
            x2 = tf.slice(bbox, [0,3], [-1,1])/img_W
            bbox = tf.concat([y1,x1,y2,x2], axis=-1)
            body_roi_list.append(tf.image.crop_and_resize(x_fg, bbox, range(batch_num), [roi_size, roi_size]))

        conv_fea_list = body_roi_list+[x_bg]

        ## Share weights for different body regions
        body_regions = tf.concat(body_roi_list, axis=0)
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = body_regions
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = slim.conv2d(body_regions, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            body_regions = body_regions + res
            if idx < repeat_num - 1:
                body_regions = slim.conv2d(body_regions, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
        body_regions = tf.reshape(body_regions, [body_regions.get_shape().as_list()[0], -1])
        body_regions = slim.fully_connected(body_regions, z_num, activation_fn=None)

        fea_list = tf.split(body_regions, bbox_num, axis=0)
        ## ROI_vis
        ROI_vis_list = tf.split(ROI_vis, bbox_num, axis=1)
        for i in range(bbox_num):
            vis_mask = ROI_vis_list[i]
            # vis_mask = tf.expand_dims(vis_mask, axis=0)
            # pdb.set_trace()
            # vis_mask = tf.Print(vis_mask, [vis_mask], 'ROI_vis vis_mask = ', summarize=10)
            vis_mask = tf.tile(vis_mask, [1, z_num])
            fea_list[i] *= vis_mask
        ## Randomly drop part
        if keep_part_prob<1.0:
            ## Randomly drop the part feature
            for i in range(bbox_num):
                vis_mask = bernoulliSample([keep_part_prob]*batch_num)
                vis_mask = tf.expand_dims(vis_mask, axis=1)
                # vis_mask = tf.Print(vis_mask, [vis_mask], 'keep_part_prob vis_mask = ', summarize=10)
                vis_mask = tf.tile(vis_mask, [1, z_num])
                fea_list[i] *= vis_mask

        ## Background branch
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x_bg
            x_bg = slim.conv2d(x_bg, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x_bg = slim.conv2d(x_bg, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x_bg = x_bg + res
            if idx < repeat_num - 1:
                x_bg = slim.conv2d(x_bg, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
        x_bg = tf.reshape(x_bg, [x_bg.get_shape().as_list()[0], -1])
        x_bg = slim.fully_connected(x_bg, z_num*4, activation_fn=None)
        # x_bg = slim.fully_connected(x_bg, z_num, activation_fn=None)

        fea_list.append(x_bg)
        fea_all = tf.concat(fea_list, axis=-1)

    variables = tf.contrib.framework.get_variables(vs)
    return fea_all, fea_list, conv_fea_list, variables


def GaussianFCRes(z_shape, out_channel, repeat_num, hidden_num, data_format, mean=0.0, stddev=0.2, activation_fn=tf.nn.relu, reuse=False, z=None):
    with tf.variable_scope("G_FC",reuse=reuse) as vs:
        if z is None:
            z = tf.random_normal(z_shape, mean, stddev, dtype=tf.float32)
        z = slim.fully_connected(z, hidden_num, activation_fn=activation_fn)
        for i in range(repeat_num):
            res = z
            z = slim.fully_connected(z, hidden_num, activation_fn=activation_fn)
            z = slim.fully_connected(z, hidden_num, activation_fn=activation_fn)
            z = res + z
        out = slim.fully_connected(z, out_channel, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def PoseEncoderFCRes(pose_rcv, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, reuse=False):
    with tf.variable_scope("G_Pose_Encoder",reuse=reuse) as vs:
        ## Encoder
        x = slim.fully_connected(pose_rcv, hidden_num, activation_fn=activation_fn)
        for i in range(repeat_num):
            res = x
            x = slim.fully_connected(x, hidden_num, activation_fn=activation_fn)
            x = slim.fully_connected(x, hidden_num, activation_fn=activation_fn)
            x = res + x
        out = slim.fully_connected(x, z_num, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def PoseDecoderFCRes(z, keypoint_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, reuse=False):
    with tf.variable_scope("G_Pose_Decoder",reuse=reuse) as vs:
        ## Decoder
        x = slim.fully_connected(z, hidden_num, activation_fn=None)
        for i in range(repeat_num):
            res = x
            x = slim.fully_connected(x, hidden_num, activation_fn=activation_fn)
            x = slim.fully_connected(x, hidden_num, activation_fn=activation_fn)
            x = res + x
        # re_pose_coord = slim.fully_connected(x, keypoint_num*2, activation_fn=tf.sigmoid)*2.0 - 1.0  ## norm to (-1, 1)
        re_pose_coord = slim.fully_connected(x, keypoint_num*2, activation_fn=None)
        re_pose_visible = slim.fully_connected(x, keypoint_num, activation_fn=tf.sigmoid)  ## norm to (0, 1)
        re_pose_visible = binaryRound(re_pose_visible)
    variables = tf.contrib.framework.get_variables(vs)
    return re_pose_coord, re_pose_visible, variables


def GeneratorCNN_ID_UAEAfterResidual(x, pose, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0, reuse=False):
    with tf.variable_scope("G",reuse=reuse) as vs:
        if pose is not None:
            if data_format == 'NCHW':
                x = tf.concat([x, pose], 1)
            elif data_format == 'NHWC':
                x = tf.concat([x, pose], 3)

        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        # x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H/2, channel_num])])
        x_shape = x.get_shape().as_list()
        x = tf.reshape(x, [x_shape[0], np.prod(x_shape[1:])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)
        if noise_dim>0:
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            z = tf.concat([z, noise], 1)

        # Decoder
        # x = slim.fully_connected(z, np.prod([min_fea_map_H, min_fea_map_H/2, hidden_num]), activation_fn=None)
        # x = reshape(x, min_fea_map_H, min_fea_map_H/2, hidden_num, data_format)
        x = slim.fully_connected(z, x_shape[1]*x_shape[2]*hidden_num, activation_fn=None)
        x = reshape(x, x_shape[1], x_shape[2], hidden_num, data_format)
        # x = reshape(x, x_shape[1], x_shape[2], x_shape[3], data_format)
        
        for idx in range(repeat_num):
            # pdb.set_trace()
            x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            res = x
            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = x.get_shape()[-1]
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = x + res
            if idx < repeat_num - 1:
                # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
                x = upscale(x, 2, data_format)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn, data_format=data_format)


        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables
