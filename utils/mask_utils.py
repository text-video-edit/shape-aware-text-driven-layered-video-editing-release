import os
import numpy as np
from path import Path
from PIL import Image
import imageio

def get_mask(img_path, output_dir, multiple_objects=False):
    mask_path = output_dir/(os.path.split(img_path)[-1].split('.')[0] + '_mask.png')
    if not os.path.isfile(mask_path):
        if multiple_objects:
            os.system('python scripts/run_mask_rcnn.py {} --output_dir {} --sum'.format(img_path, output_dir))
        else:
            os.system('python scripts/run_mask_rcnn.py {} --output_dir {}'.format(img_path, output_dir))
    return imageio.imread(mask_path) 

def find_mask_border(mask, pad=0):
    '''
    Input:
        mask: [H, W]
    Return:
        t, b, l, r
    '''

    H, W = mask.shape
    t = 0
    while t < H and mask[t, :].sum() < 1:
        t += 1
    b = H - 1
    while b >= 0 and mask[b, :].sum() < 1:
        b -= 1
    l = 0
    while l < W and mask[:, l].sum() < 1:
        l += 1
    r = W - 1
    while r >= 0 and mask[:, r].sum() < 1:
        r -= 1

    t = max(0, t - pad); b = min(H - 1, b + pad); l = max(0, l - pad); r = min(W - 1, r + pad)
    return t, b, l, r

def get_crop_area(mask, shape, pad=0, keep_aspect=True):
    H, W = shape
    h, w = mask.shape[-2:]
    t, b, l, r = find_mask_border(mask.squeeze(0), pad=pad)
    if keep_aspect:
        h_ = b - t
        w_ = r - l
        if h_ / w_ > H / W:
            w1 = h_ * W / H
            l = l - (w1 - w_) * 0.5 
            r = r + (w1 - w_) * 0.5
        else:
            h2 = w_ * H / W
            t = t - (h2 - h_) * 0.5
            b = b + (h2 - h_) * 0.5
    t = max(0, t)
    b = min(h - 1, b)
    l = max(0, l)
    r = min(w - 1, r)
    t /= (h - 1)
    b /= (h - 1)
    l /= (w - 1)
    r /= (w - 1)

    return [t, b, l, r] 

