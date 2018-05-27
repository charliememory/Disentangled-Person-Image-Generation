from __future__ import print_function

import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
import tensorflow as tf

####################### model related #########################
def var_filter_by_exclude(var_list, exclude_scopes=[], Print=False):    
    # exclude_scopes=['InceptionV1/Logits', 'InceptionV1/AuxLogits', 'Ver', 'Cla', 'Aux','CMC', 'Base']
    exclusions = [scope.strip() for scope in exclude_scopes]

    variables_to_restore = []
    for var in var_list:
        if Print:
            print('mlq --- variable:')
            print(var.op.name)
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
            if Print:
                print('restore')
        else:
            if Print:
                print('excluded')
    return variables_to_restore

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (int(h*scale), int(w*scale)), data_format)

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high



###################### utils_wgan ##########################
def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm


def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel


################################################
def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir') or config.model_dir is None:
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

##################### Pose/Mask/Sparse #####################
import scipy.io
import scipy.stats
import skimage.morphology
from skimage.morphology import square, dilation, erosion
from PIL import Image

def coord2channel_simple_MASK_RC00(coords, keypoint_num=18, is_normalized=True ,img_H=128, img_W=64, MASK_RC00=True):
    ## MASK_RC00: if the keypoint is not detected, the RC is [0,0], 
    ## we will set the [0,0] of all channels to -1
    print('######coord2channel_simple#####')
    batch_size = coords.get_shape().as_list()[0]
    coords = tf.reshape(coords, [batch_size, keypoint_num, 2])
    R = tf.slice(coords, [0,0,0], [-1,-1,1])
    C = tf.slice(coords, [0,0,1], [-1,-1,1])
    if is_normalized:
        R = (R + 1)/2.0*img_H ## reverse norm to 256,256 
        R = tf.maximum(tf.zeros_like(R), R) ## keep the coords in image
        R = tf.minimum(tf.ones_like(R)*img_H-1, R) ## keep the coords in image
        C = (C + 1)/2.0*img_W ## reverse norm to 256,256 
        C = tf.maximum(tf.zeros_like(C), C) ## keep the coords in image
        C = tf.minimum(tf.ones_like(C)*img_W-1, C) ## keep the coords in image
    coords = tf.concat([R,C], axis=-1)

    ## Note: reshape starts from the last axis
    coords = tf.to_int32(coords)
    ## coords stores x,y
    R = tf.slice(coords, [0,0,0], [-1,-1,1])
    R = tf.reshape(R, [-1])
    C = tf.slice(coords, [0,0,1], [-1,-1,1])
    C = tf.reshape(C, [-1])

    batch_size = coords.get_shape().as_list()[0]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    B = tf.tile(batch_idx, (1, keypoint_num))
    B = tf.reshape(B, [-1])     
    kp_idx = tf.range(0, keypoint_num)
    K = tf.tile(kp_idx, [batch_size])

    indices = tf.stack([B, R, C, K], axis=1)
    updates = 2*tf.ones([batch_size*keypoint_num]) ## first [0,2], then reduce to [-1,1]
    shape=tf.constant([batch_size, img_H, img_W, keypoint_num])
    landChannel = tf.scatter_nd(indices=indices, updates=updates, shape=shape)
    if MASK_RC00:
        mask = np.ones([batch_size, img_H, img_W, keypoint_num])
        mask[:, 0, 0, :] *= 0 
        tf_mask = tf.constant(mask)
        landChannel = tf.multiply(landChannel, tf.cast(tf_mask, landChannel.dtype))
    landChannel = landChannel - 1 ## first [0,2], then reduce to [-1,1]
    return landChannel

def coord2channel_simple_rcv(RCV, keypoint_num=18, is_normalized=True ,img_H=128, img_W=64):
    ## MASK_RC00: if the keypoint is not detected, the RC is [0,0], 
    ## we will set the [0,0] of all channels to -1
    print('######coord2channel_simple#####')
    batch_size = RCV.get_shape().as_list()[0]
    RCV = tf.reshape(RCV, [batch_size, keypoint_num, 3])
    R = tf.slice(RCV, [0,0,0], [-1,-1,1])
    C = tf.slice(RCV, [0,0,1], [-1,-1,1])
    V = tf.slice(RCV, [0,0,2], [-1,-1,1])

    # R = tf.Print(R, [R], 'R before = ', summarize=10)
    if is_normalized:
        R = (R + 1)/2.0*img_H ## reverse norm to 256,256 
        R = tf.maximum(tf.zeros_like(R), R) ## keep the coords in image
        R = tf.minimum(tf.ones_like(R)*img_H-1, R) ## keep the coords in image
        C = (C + 1)/2.0*img_W ## reverse norm to 256,256 
        C = tf.maximum(tf.zeros_like(C), C) ## keep the coords in image
        C = tf.minimum(tf.ones_like(C)*img_W-1, C) ## keep the coords in image
    coords = tf.concat([R,C], axis=-1)
    # R = tf.Print(R, [R], 'R after = ', summarize=10)

    ## Note: reshape starts from the last axis
    coords = tf.to_int32(coords)
    ## coords stores x,y
    R = tf.slice(coords, [0,0,0], [-1,-1,1])
    R = tf.reshape(R, [-1])
    C = tf.slice(coords, [0,0,1], [-1,-1,1])
    C = tf.reshape(C, [-1])

    batch_size = coords.get_shape().as_list()[0]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    B = tf.tile(batch_idx, (1, keypoint_num))
    B = tf.reshape(B, [-1])     
    kp_idx = tf.range(0, keypoint_num)
    K = tf.tile(kp_idx, [batch_size])

    indices = tf.stack([B, R, C, K], axis=1)
    updates = 2*tf.ones([batch_size*keypoint_num]) ## first [0,2], then reduce to [-1,1]
    shape=tf.constant([batch_size, img_H, img_W, keypoint_num])
    landChannel = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

    V = tf.tile(V, [1,1,img_H*img_W])
    V = tf.reshape(V, [batch_size, keypoint_num, img_H, img_W])
    V = nchw_to_nhwc(V)

    landChannel = landChannel*V

    landChannel = landChannel - 1 ## first [0,2], then reduce to [-1,1]
    return landChannel

def tf_poseInflate(G_pose, keypoint_num, radius=4, img_H=128, img_W=64):
    # def transfer_pose_rcv(pose_rcv_batch, x_offset, y_offset):
    #     return pose_rcv_batch
    def _poseInflate(pose_channel, x_offset, y_offset, radius):
        pose_channel = tf.image.pad_to_bounding_box(pose_channel, radius, radius, img_H+radius*2, img_W+radius*2)
        pose_channel = tf.image.crop_to_bounding_box(pose_channel, x_offset+radius, y_offset+radius, img_H, img_W)
        return pose_channel

    G_pose = (G_pose+1)/2  ## Change [-1,1] to [0,1]
    G_pose_inflated = G_pose

    for x_offset in [-4,4]:
        for y_offset in [0]:
            G_pose_inflated += _poseInflate(G_pose, x_offset, y_offset, radius)
    for x_offset in [-3,3]:
        for y_offset in range(-2,3):
            G_pose_inflated += _poseInflate(G_pose, x_offset, y_offset, radius)
    for x_offset in [-2,2]:
        for y_offset in range(-3,4):
            G_pose_inflated += _poseInflate(G_pose, x_offset, y_offset, radius)
    for x_offset in [-1,1]:
        for y_offset in range(-3,4):
            G_pose_inflated += _poseInflate(G_pose, x_offset, y_offset, radius)
    for x_offset in [0]:
        for y_offset in range(-4,5):
            G_pose_inflated += _poseInflate(G_pose, x_offset, y_offset, radius)

    G_pose_inflated = tf.minimum(G_pose_inflated, 1)  ## Change [0,1+] to [0,1]
    G_pose_inflated = G_pose_inflated*2-1  ## Change [0,1] to [-1,1]
    return G_pose_inflated

def py_poseInflate(pose_rcv_batch, is_normalized=True, radius=4, img_H=128, img_W=64):
    def py_fillMatrix(dense, r, c, k, radius, img_H, img_W):
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                distance = np.sqrt(float(i**2+j**2))
                if r+i>=0 and r+i<img_H and c+j>=0 and c+j<img_W and distance<=radius:
                    dense[int(r+i),int(c+j),k] = 1
        return dense

    batch_size, keypoint_num, _ = pose_rcv_batch.shape
    pose_dense_batch = np.zeros([batch_size, img_H, img_W, keypoint_num])
    for b in range(batch_size):
        pose_dense = np.zeros([img_H, img_W, keypoint_num])
        for k in range(keypoint_num):
            r, c, v = pose_rcv_batch[b,k,:]
            if is_normalized:
                r = (r + 1)/2.0*img_H ## reverse norm to 256,256 
                r = np.maximum(0, r) ## keep the coords in image
                r = np.minimum(img_H-1, r) ## keep the coords in image
                c = (c + 1)/2.0*img_W ## reverse norm to 256,256 
                c = np.maximum(0, c) ## keep the coords in image
                c = np.minimum(img_W-1, c) ## keep the coords in image
            if v:
                pose_dense = py_fillMatrix(pose_dense, r, c, k, radius, img_H, img_W)
        pose_dense_batch[b,:,:,:] = pose_dense
    pose_dense_batch = pose_dense_batch.astype(np.float)*2.0 - 1.0 ## norm to [-1,1]
    return pose_dense_batch


def _getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):
    ## MSCOCO Pose part_str = [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    # find connection in the specified sequence, center 29 is in the position 15
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18], [3,17], [6,18]]
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # , [9,12]
    # limbSeq = [[3,4], [4,5], [6,7], [7,8], [9,10], \
    #            [10,11], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18]] # 
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                         [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                         [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]] #
    indices = []
    values = []
    for limb in limbSeq:
        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        if 0!=len(p0) and 0!=len(p1):
            r0 = p0[0][1]
            c0 = p0[0][0]
            r1 = p1[0][1]
            c1 = p1[0][0]
            ind, val = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
        
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            # sampleN = 0
            if sampleN>1:
                for i in xrange(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(_sparse2dense(indices, values, shape))
    ## TODO
    # im = Image.fromarray((dense*255).astype(np.uint8))
    # im.save('xxxxx.png')
    # pdb.set_trace()
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))
    return dense


Ratio_0_4 = 1.0/scipy.stats.norm(0, 4).pdf(0)
Gaussian_0_4 = scipy.stats.norm(0, 4)
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)
                elif 'Gaussian'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    if 4==var:
                        values.append( Gaussian_0_4.pdf(distance) * Ratio_0_4  )
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values

def _getSparsePose(peaks, height, width, channel, radius=4, var=4, mode='Solid'):
    indices = []
    values = []
    for k in range(len(peaks)):
        p = peaks[k]
        if 0!=len(p):
            r = p[0][1]
            c = p[0][0]
            ind, val = _getSparseKeypoint(r, c, k, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
    shape = [height, width, channel]
    return indices, values, shape

def _oneDimSparsePose(indices, shape):
    ind_onedim = []
    for ind in indices:
        # idx = ind[2]*shape[0]*shape[1] + ind[1]*shape[0] + ind[0]
        idx = ind[0]*shape[2]*shape[1] + ind[1]*shape[2] + ind[2]
        ind_onedim.append(idx)
    shape = np.prod(shape)
    return ind_onedim, shape

def _sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = values[i]
    return dense

def _get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]
            # for s in subset:
            #   if s > -1:
            #     cnt += 1
            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx>=0:
            peaks = []
            cand_id_list = subsets[valid_idx][:18]

            for ap in all_peaks:
                valid_p = []
                for p in ap:
                    if p[-1] in cand_id_list:
                        valid_p = p
                if len(valid_p)>0: # use the same structure with all_peaks
                    peaks.append([(valid_p)])
                else:
                    peaks.append([])
            return peaks
        else:
            return None
    except:
        # pdb.set_trace()
        return None

import matplotlib.pyplot as plt 
import scipy.misc
def _visualizePose(pose, img):
    # pdb.set_trace()
    if 3==len(pose.shape):
        pose = pose.max(axis=-1, keepdims=True)
        pose = np.tile(pose, (1,1,3))
    elif 2==len(pose.shape):
        pose = np.expand_dims(pose, -1)
        pose = np.tile(pose, (1,1,3))

    imgShow = ((pose.astype(np.float)+1)/2.0*img.astype(np.float)).astype(np.uint8)
    plt.imshow(imgShow)
    plt.show()