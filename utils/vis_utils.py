# Taken from MF-TAPNet official repository.
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2

def draw_plus(image, center, color=(0, 255, 0), size=5, thickness=1):
    """
    Draws a plus symbol at the specified center on the image.

    Args:
    - image (numpy.ndarray): The image on which to draw the cross.
    - center (tuple): The (x, y) coordinates for the center of the cross.
    - size (int, optional): Length of each line in the cross. Default is 50.
    - color (tuple, optional): Color of the cross in BGR format. Default is green (0, 255, 0).
    - thickness (int, optional): Thickness of the cross lines. Default is 2.

    Returns:
    - image (numpy.ndarray): The image with the cross drawn on it.
    """
    if np.isnan(center).any():
        return image
    # Draw a vertical line (x remains the same, y varies)
    cv2.line(image, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)
    # Draw a horizontal line (y remains the same, x varies)
    cv2.line(image, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    return image

def mask_overlay(image, mask, color=(0, 255, 0), wt=0.5):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, wt, image, 1-wt, 0.0)
    img = image.copy() 
    ind = mask[:,:,1] > 0 
    img[ind] = weighted_sum[ind]
    return img

def flow_to_arrow(flow_uv, positive=True):
    '''
    Expects a two dimensional flow image of shape [H,W,2]. Author: Keyun Cheng, 2019-01-01.

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return: flow image shown in arrow
    '''
    h, w = flow_uv.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = np.rint(x + flow_uv[:,:,0]).astype(dtype=np.int64)
    new_y = np.rint(y + flow_uv[:,:,1]).astype(dtype=np.int64)
    # clip to the boundary
    new_x = np.clip(new_x, 0, w)
    new_y = np.clip(new_y, 0, h)
    # empty image
    coords_origin = np.array([x.flatten(), y.flatten()]).T
    coords_new = np.array([new_x.flatten(), new_y.flatten()]).T

    flow_arrow = np.ones((h, w, 3), np.uint8) * 255
    for i in range(0, len(coords_origin), 1000):
        if positive:
            cv2.arrowedLine(flow_arrow, tuple(coords_origin[i]), tuple(coords_new[i]), (255, 0, 0), 2)
        else:
            cv2.arrowedLine(flow_arrow, tuple(coords_new[i]), tuple(coords_origin[i]), (255, 0, 0), 2)
    return flow_arrow

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2 * (ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,colorwheel.shape[1] - i - 1]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)
        