import numpy as np
from scipy import ndimage

import tensorflow as tf

from kinect_init import *


def processPixelStage2(m):
    dtype = tf.float32
    PI = 3.14159265358979323846
    # m is (None, 424, 512, 9)
    # the first three is measurement a
    # the second three is measurement b
    # the third three is amplitude
    tmp0 = tf.atan2(m[:, :, 3:6], m[:, :, 0:3])
    flg = tf.cast(tf.less(tmp0, 0.0), dtype)
    tmp0 = flg * (tmp0 + PI * 2) + (1 - flg) * tmp0

    tmp1 = tf.sqrt(m[:, :, 0:3] ** 2 + m[:, :, 3:6] ** 2) * prms['ab_multiplier']

    ir_sum = tf.reduce_sum(tmp1, -1)

    # disable disambiguation
    ir_min = tf.reduce_min(tmp1, -1)

    # phase mask
    phase_msk1 = tf.cast( \
        tf.greater(ir_min, prms['individual_ab_threshold']),
        dtype=dtype
    )
    phase_msk2 = tf.cast( \
        tf.greater(ir_sum, prms['ab_threshold']),
        dtype=dtype
    )
    phase_msk_t = phase_msk1 * phase_msk2

    # compute phase
    t0 = tmp0[:, :, 0] / (2.0 * PI) * 3.0
    t1 = tmp0[:, :, 1] / (2.0 * PI) * 15.0
    t2 = tmp0[:, :, 2] / (2.0 * PI) * 2.0

    t5 = tf.floor((t1 - t0) * 0.3333333 + 0.5) * 3.0 + t0
    t3 = t5 - t2
    t4 = t3 * 2.0

    c1 = tf.cast(tf.greater(t4, -t4), dtype=dtype)
    f1 = c1 * 2.0 + (1 - c1) * (-2.0)
    f2 = c1 * 0.5 + (1 - c1) * (-0.5)
    t3 = t3 * f2
    t3 = (t3 - tf.floor(t3)) * f1

    c2 = tf.cast(tf.less(0.5, tf.abs(t3)), dtype=dtype) * \
         tf.cast(tf.less(tf.abs(t3), 1.5), dtype=dtype)
    t6 = c2 * (t5 + 15.0) + (1 - c2) * t5
    t7 = c2 * (t1 + 15.0) + (1 - c2) * t1
    t8 = (tf.floor((t6 - t2) * 0.5 + 0.5) * 2.0 + t2) * 0.5

    t6 /= 3.0
    t7 /= 15.0

    # transformed phase measurements (they are transformed and divided
    # by the values the original values were multiplied with)
    t9 = t8 + t6 + t7
    t10 = t9 / 3.0  # some avg

    t6 = t6 * 2.0 * PI
    t7 = t7 * 2.0 * PI
    t8 = t8 * 2.0 * PI

    t8_new = t7 * 0.826977 - t8 * 0.110264
    t6_new = t8 * 0.551318 - t6 * 0.826977
    t7_new = t6 * 0.110264 - t7 * 0.551318

    t8 = t8_new
    t6 = t6_new
    t7 = t7_new

    norm = t8 ** 2 + t6 ** 2 + t7 ** 2
    mask = tf.cast(tf.greater(t9, 0.0), dtype)
    t10 = t10 * mask

    slope_positive = float(0 < prms['ab_confidence_slope'])

    ir_min_ = tf.reduce_min(tmp1, -1)
    ir_max_ = tf.reduce_max(tmp1, -1)

    ir_x = slope_positive * ir_min_ + (1 - slope_positive) * ir_max_

    ir_x = tf.math.log(ir_x)
    ir_x = (ir_x * prms['ab_confidence_slope'] * 0.301030 + prms['ab_confidence_offset']) * 3.321928
    ir_x = tf.exp(ir_x)
    ir_x = tf.maximum(prms['min_dealias_confidence'], ir_x)
    ir_x = tf.minimum(prms['max_dealias_confidence'], ir_x)
    ir_x = ir_x ** 2

    mask2 = tf.cast(tf.greater(ir_x, norm), dtype)

    t11 = t10 * mask2

    mask3 = tf.cast( \
        tf.greater(prms['max_dealias_confidence'] ** 2, norm),
        dtype
    )
    t10 = t10 * mask3
    phase = t11

    # mask out dim regions
    phase = phase * phase_msk_t

    # phase to depth mapping
    zmultiplier = z_table
    xmultiplier = x_table

    phase_msk = tf.cast(tf.less(0.0, phase), dtype)
    phase = phase_msk * (phase + prms['phase_offset']) + (1 - phase_msk) * phase

    depth_linear = zmultiplier * phase
    max_depth = phase * prms['unambiguous_dist'] * 2

    cond1 = tf.cast(tf.less(0.0, depth_linear), dtype) * \
            tf.cast(tf.less(0.0, max_depth), dtype)

    xmultiplier = (xmultiplier * 90) / (max_depth ** 2 * 8192.0)

    depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)

    depth_fit = tf.maximum(depth_fit, 0.0)
    depth = cond1 * depth_fit + (1 - cond1) * depth_linear

    return depth


def tof_net_func(x):
    depth_outs = processPixelStage2(x)
    return depth_outs


def distance_transform(depth, mask):
    depth = np.array(depth, dtype=float)
    mask = np.array(mask, dtype=bool)
    
    if depth.shape != mask.shape:
        raise ValueError
    
    out_depth = depth.copy()
    
    known_values = depth[~mask]
    
    if len(known_values) == 0:
        return depth
    
    _, indices = ndimage.distance_transform_edt(
        mask,
        return_indices=True  
    )

    out_depth[mask] = depth[indices[0][mask], indices[1][mask]]
    
    return out_depth


def iq2depth(pred, amplitudes):
    order = [2, 5, 8]
    nimg_iq = np.transpose(pred, (1, 2, 0))
    x = np.stack((nimg_iq[:, :, 3],
                    nimg_iq[:, :, 1],
                    nimg_iq[:, :, 5],
                    nimg_iq[:, :, 2],
                    nimg_iq[:, :, 0],
                    nimg_iq[:, :, 4]), axis=2)
    
    for i in order:
        if i == 2 or i == 5 or i == 8:
            x = np.concatenate((x, amplitudes[:, :, i:i+1]), axis=2)
    
    depth = tof_net_func(x)
    with tf.compat.v1.Session():
        depth = depth.numpy() / 1e3
    depth = np.nan_to_num(depth, 0)
    mask = ((depth < 0.1) | (depth > 5)).astype(np.uint8)

    return distance_transform(depth, mask)
