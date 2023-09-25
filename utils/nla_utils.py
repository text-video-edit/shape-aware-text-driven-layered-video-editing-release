import numpy as np
import torch
from .warp_utils import *
from .mask_utils import *

def crop_texture(texture, uvs, masks=None, pad=50):
    x_min, x_max, y_min, y_max = 1, -1, 1, -1
    N, h, w, _ = uvs.shape
    _, _, H, W = texture.shape
    
    if masks is not None:
        for i in range(N):
            if masks[i].sum() < 1: continue
            x_min = min(x_min, uvs[i, :, :, 0][masks[i, 0] > 0.5].min())
            x_max = max(x_max, uvs[i, :, :, 0][masks[i, 0] > 0.5].max())
            y_min = min(y_min, uvs[i, :, :, 1][masks[i, 0] > 0.5].min())
            y_max = max(y_max, uvs[i, :, :, 1][masks[i, 0] > 0.5].max())
    else:
        x_min = uvs[..., 0].min()
        x_max = uvs[..., 0].max()
        y_min = uvs[..., 1].min()
        y_max = uvs[..., 1].max()

    l = max(int((x_min + 1) * 0.5 * (W - 1)) - pad, 0)
    r = min(int((x_max + 1) * 0.5 * (W - 1)) + pad, W - 1)
    t = max(int((y_min + 1) * 0.5 * (H - 1)) - pad, 0)
    b = min(int((y_max + 1) * 0.5 * (H - 1)) + pad, H - 1)

    texture_crop = texture[..., t:b, l:r]
    return texture_crop, (t/(H-1), b/(H-1), l/(W-1), r/(W-1))

def crop_uv(uvs, masks, pad=50):
    uvs_crop = uvs.clone()
    crop_areas = []
    for i in range(N):
        t, b, l, r = get_crop_area(masks[i], (h, w), pad=pad)
        crop_areas.append((t, b, l, r))

        t = int(t * (h -1))
        b = int(b * (h -1))
        l = int(l * (w -1))
        r = int(r * (w -1))
        uvs_crop[[i]] = resize_uv(uvs_crop[[i]][:, t:b, l:r, :], (h, w))
    return uvs_crop, crop_areas
