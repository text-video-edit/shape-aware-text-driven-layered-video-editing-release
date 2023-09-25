import torch
import torch.nn as nn
import torch.nn.functional as F
import oflibpytorch as of
import imageio
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_uv_mask(uv):
    mask = (uv[..., 0] >= -1) * (uv[..., 0] <= 1) * (uv[..., 1] >= -1) * (uv[..., 1] <= 1)
    mask = mask.float().unsqueeze(1)
    return mask

def get_meshgrid(h, w, normalize=False):
    xs = torch.linspace(0, w - 1, w)
    ys = torch.linspace(0, h - 1, h)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    meshgrid = torch.stack((xx, yy), 0).unsqueeze(0)
    if normalize:
        meshgrid[:, 0] = meshgrid[:, 0] / (w - 1) * 2 - 1
        meshgrid[:, 1] = meshgrid[:, 1] / (h - 1) * 2 - 1
    return meshgrid

def get_text_criterion(cfg):
    if cfg["text_criterion"] == "spherical":
        text_criterion = spherical_dist_loss
    elif cfg["text_criterion"] == "cosine":
        text_criterion = cosine_loss
    else:
        return NotImplementedError("text criterion [%s] is not implemented", cfg["text_criterion"])
    return text_criterion

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return ((x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)).mean()

def cosine_loss(x, y, scaling=1.2):
    return scaling * (1 - F.cosine_similarity(x, y).mean())

def get_screen_template():
    return [
        "{} over a green screen.",
        "{} in front of a green screen.",
    ]

def get_augmentations_template():
    templates = [
        "photo of {}.",
        "high quality photo of {}.",
        "a photo of {}.",
        "the photo of {}.",
        "image of {}.",
        "an image of {}.",
        "high quality image of {}.",
        "a high quality image of {}.",
        "an undistorted image of {}.",
        "a real image of {}.",
        "an undistorted photo of {}.",
        "a real photo of {}.",
        "the {}.",
        "a {}.",
        "{}.",
        "{}",
        #"{}!",
        #"{}...",
    ]
    return templates

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

def grid_uv2texture(uv_maps, texture):

    uv_maps_ds = F.interpolate(
            uv_maps.permute(0, 3, 1, 2), 
            scale_factor=0.5, 
            mode='bilinear').permute(0, 2, 3, 1)

    H, W, _ = texture.shape
    N, h, w, _ = uv_maps_ds.shape
    num_vtx_h = 20
    num_vtx_w = 20
    ys = torch.linspace(0, h - 1, num_vtx_h)
    xs = torch.linspace(0, w - 1, num_vtx_w)
    xx, yy = torch.meshgrid(xs, ys, indexing='ij')
    grids = torch.stack((xx, yy), -1).view(-1, 2).long()
    
    uv_maps_ds[..., 0] = (uv_maps_ds[..., 0] + 1) * 0.5 * W
    uv_maps_ds[..., 1] = (uv_maps_ds[..., 1] + 1) * 0.5 * H
    
    vtx_uv_maps = uv_maps_ds[:, grids[:, 1], grids[:, 0]]
    vtx_uv_maps = vtx_uv_maps.view(N, num_vtx_h, num_vtx_w, 2)
    return vtx_uv_maps

def grid_texture2uv(uv_maps, texture):
    N, h, w, _ = uv_maps.shape
    H, W, _ = texture.shape
    
    num_vtx_h = 20
    num_vtx_w = 20
    
    minx = uv_maps[..., 0].min()
    maxx = uv_maps[..., 0].max()
    miny = uv_maps[..., 1].min()
    maxy = uv_maps[..., 1].max()

    ys = torch.linspace(miny, maxy, num_vtx_h).to(device)
    xs = torch.linspace(minx, maxx, num_vtx_w).to(device)
    
    xx, yy = torch.meshgrid(xs, ys, indexing='ij')
    grids = torch.stack((xx, yy), -1)
    
    correspondences = torch.ones(N, num_vtx_h, num_vtx_w, 2).to(device).long() * -5
    
    for j in range(num_vtx_h):
        for i in range(num_vtx_w):
            diff = uv_maps.view(N, h * w, 2) - grids[None, j, i]
            diff2 = torch.sqrt((diff * diff).sum(-1))
            mins, indices = torch.min(diff2, dim=1)
            valid_mask = mins < 1e-2
            indices_y = indices // w
            indices_x = indices % w
            
            correspondences[valid_mask, j, i, 0] = indices_x[valid_mask]
            correspondences[valid_mask, j, i, 1] = indices_y[valid_mask]
    
    return correspondences

def warp_uv(target_uv, transform_uv):
    '''
    Input tensors:
        target_uv: [N, H, W, 2]
        transform_uv: [N, H, W, 2]
    '''
    mask = get_uv_mask(transform_uv)
    warped_uv = backward_warp(target_uv.permute(0, 3, 1, 2), transform_uv)
    warped_uv = warped_uv * mask + torch.ones(warped_uv.shape).to(warped_uv.device) * (1 - mask) * -2
    return warped_uv.permute(0, 2, 3, 1)

def forward_warp(image, uv, fill=0):
    '''
    Input tensors:
        image: [N, 3, H, W]
        uv (forward): [N, H, W, 2]
    '''
    uv_inv = inverse_uv(uv.shape[1:-1], uv)
    mask = ((uv_inv[..., 0] >= -1) * (uv_inv[..., 0] <= 1) * (uv_inv[..., 1] >= -1) * (uv_inv[..., 1] <= 1)).float()
    image_warped = F.grid_sample(image, uv_inv, mode='bilinear')
    if fill != 0:
        image_warped = image_warped * mask[:, None, :, :] + \
                        torch.ones(image_warped.shape).to(image_warped.device) * (1 - mask[:, None, :, :]) * fill
    return image_warped


def inverse_uv(target_shape, _uv_maps):
    # code from https://github.com/yaochih/Deep3D-Stabilizer-release
    # given UV from image to texture, return the inverse UV from texture to image
    '''
    Input tensors:
        target_shape: (H, W)
        uv_maps: [b, h, w, 2], range [-1, 1]
    Return:
        inv_uv_maps: [b, H, W, 2], range [-1, 1]
    '''
    uv_maps = _uv_maps.clone()
    H, W = target_shape
    N, h, w, _ = uv_maps.shape #forward_flows.size()
    
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).float()
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).float()
    pixel_map = torch.cat((j_range, i_range), 0).permute(1, 2, 0).unsqueeze(0).to(uv_maps.device)
    
    sh, sw = 3, 3
    idxs = {i * sw + j: [] for i in range(sh) for j in range(sw)}
    for i in range(h):
        for j in range(w):
            key = ((i % sh) * sw) + j % sw
            idxs[key].append(i * w + j)
    idx_set = [torch.Tensor(v).long().to(uv_maps.device) for v in idxs.values()]

    uv_maps[..., 0] = (uv_maps[..., 0] + 1) * 0.5 * W
    uv_maps[..., 1] = (uv_maps[..., 1] + 1) * 0.5 * H

    x = uv_maps[..., 0].view(N, -1)
    y = uv_maps[..., 1].view(N, -1)
    l = torch.floor(x); r = l + 1
    t = torch.floor(y); b = t + 1

    mask = (l >= 0) * (t >= 0) * (r < W) * (b < H)
    l *= mask; r *= mask; t *= mask; b *= mask
    x *= mask; y *= mask
    w_rb = torch.abs(x - l + 1e-3) * torch.abs(y - t + 1e-3)
    w_rt = torch.abs(x - l + 1e-3) * torch.abs(b - y + 1e-3)
    w_lb = torch.abs(r - x + 1e-3) * torch.abs(y - t + 1e-3)
    w_lt = torch.abs(r - x + 1e-3) * torch.abs(b - y + 1e-3)
    l = l.long(); r = r.long(); t = t.long(); b = b.long()

    weight_maps = torch.zeros(N, H, W).to(uv_maps.device).double()
    grid_x = pixel_map[..., 0].view(-1).long()
    grid_y = pixel_map[..., 1].view(-1).long()

    for i in range(N):
        for j in idx_set:
            weight_maps[i, t[i, j], l[i, j]] += w_lt[i, j]
            weight_maps[i, t[i, j], r[i, j]] += w_rt[i, j]
            weight_maps[i, b[i, j], l[i, j]] += w_lb[i, j]
            weight_maps[i, b[i, j], r[i, j]] += w_rb[i, j]
    
    inv_uv_maps = torch.zeros(N, H, W, 2).to(uv_maps.device)
    for i in range(N):
        for j in idx_set:
            for c in range(2):
                inv_uv_maps[i, t[i, j], l[i, j], c] += pixel_map[0, :, :, c].view(-1)[j] * w_lt[i, j]
                inv_uv_maps[i, t[i, j], r[i, j], c] += pixel_map[0, :, :, c].view(-1)[j] * w_rt[i, j]
                inv_uv_maps[i, b[i, j], l[i, j], c] += pixel_map[0, :, :, c].view(-1)[j] * w_lb[i, j]
                inv_uv_maps[i, b[i, j], r[i, j], c] += pixel_map[0, :, :, c].view(-1)[j] * w_rb[i, j]

    inv_uv_maps /= weight_maps[:, :, :, None]

    inv_uv_maps[torch.isinf(inv_uv_maps)] = 0
    inv_uv_maps[torch.isnan(inv_uv_maps)] = 0

    inv_uv_maps[weight_maps == 0] = -2

    inv_uv_maps[..., 0] = inv_uv_maps[..., 0] / w * 2 - 1
    inv_uv_maps[..., 1] = inv_uv_maps[..., 1] / h * 2 - 1

    uv_maps[..., 0] = uv_maps[..., 0] / W * 2 - 1
    uv_maps[..., 1] = uv_maps[..., 1] / H * 2 - 1

    return inv_uv_maps


def oflib_inverse_uv(target_shape, uv):
    _, h, w, _ = uv.shape
    meshgrid = get_meshgrid(h, w, normalize=True).to(uv.device)
    inverse_uv = oflib_forward_warp(meshgrid, uv)
    inverse_uv = F.interpolate(inverse_uv, target_shape, mode='bilinear')
    return inverse_uv.permute(0, 2, 3, 1)

def oflib_forward_warp(image, uv, fill=0):
    h, w = uv.shape[1:-1]
    uv_ = uv.clone()
    uv_[..., 0] = (uv_[..., 0] + 1) / 2 * (w - 1)
    uv_[..., 1] = (uv_[..., 1] + 1) / 2 * (h - 1)
    uv_ -= get_meshgrid(h, w).permute(0, 2, 3, 1).to(uv.device)
    
    warp_image, mask = of.apply_s_flow(uv_.permute(0, 3, 1, 2), image)
    mask = mask.float()[:, None, :, :]
    warp_image = warp_image * mask + torch.ones(warp_image.shape).to(uv.device) * fill * (1 - mask)
    return warp_image


def get_homo_warp_grid(homo, target_shape):
    N, _, _ = homo.shape
    _, _, h, w = target_shape
    grid = get_meshgrid(h, w, normalize=True).to(homo.device)
    grid = torch.cat((grid, torch.ones(grid[:, :1].shape).to(grid.device)), 1).repeat(N, 1, 1, 1)
    
    warp_grid = (homo @ grid.reshape(N, 3, -1)).reshape(N, 3, h, w).permute(0, 2, 3, 1)
    warp_grid[..., 0] /= warp_grid[..., -1]
    warp_grid[..., 1] /= warp_grid[..., -1]
    warp_grid = warp_grid[..., :2]
    return warp_grid

def backward_warp(image, uv, fill=0, padding='zeros'):
    mask = get_uv_mask(uv).to(uv.device)
    warped_image = F.grid_sample(image, uv, mode='bilinear', padding_mode=padding)
    warped_image = warped_image * mask + torch.ones(warped_image.shape).to(image.device) * (1 - mask) * fill
    return warped_image

def backward_warp_nn(image, uv, fill=0, padding='zeros'):
    mask = get_uv_mask(uv).to(uv.device)
    warped_image = F.grid_sample(image, uv, mode='nearest', padding_mode=padding)
    warped_image = warped_image * mask + torch.ones(warped_image.shape).to(image.device) * (1 - mask) * fill
    return warped_image

def resize_uv(uv, shape):
    return F.interpolate(uv.permute(0, 3, 1, 2), shape, mode='bilinear').permute(0, 2, 3, 1)

def gaussian_fn(M, std):
    n = torch.arange(M) - (M - 1.) / 2.
    sig2 = std * std * 2
    w = torch.exp(-n ** 2 / sig2)
    return w / w.sum()

def scale_to_crop(uv, crop_area):
    t, b, l, r = crop_area
    uv_ = uv.clone()
    uv_[..., 0] = (uv_[..., 0] + 1) * 0.5 
    uv_[..., 1] = (uv_[..., 1] + 1) * 0.5
    
    uv_[..., 0] = (uv_[..., 0] - l) / (r - l) * 2 - 1
    uv_[..., 1] = (uv_[..., 1] - t) / (b - t) * 2 - 1
    return uv_

def rescale_from_crop(uv, crop_area):
    t, b, l, r = crop_area
    uv_ = uv.clone()
    uv_[..., 0] = (uv_[..., 0] + 1) * 0.5
    uv_[..., 1] = (uv_[..., 1] + 1) * 0.5
    uv_[..., 0] = (uv_[..., 0] * (r - l) + l) * 2 - 1
    uv_[..., 1] = (uv_[..., 1] * (b - t) + t) * 2 - 1
    return uv_


