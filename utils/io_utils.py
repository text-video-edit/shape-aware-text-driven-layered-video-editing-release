import os
import imageio
import torch
import torch.nn.functional as F
import numpy as np
from .warp_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_images(frame_list, size=None):
    frames = []
    for frame_path in frame_list:
        img = torch.FloatTensor(imageio.imread(frame_path)) / 255.
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        if size is not None:
            img = F.interpolate(img, size, mode='bilinear')
        frames.append(img)
    frames = torch.cat(frames, 0)
    return frames

def get_correspondence(edit_dir, shape=None):
    semantic_corr = torch.FloatTensor(np.load(edit_dir/'semantic_correspondence_crop.npy')).unsqueeze(0)
    if shape is not None:
        semantic_corr = F.interpolate(semantic_corr.permute(0, 3, 1, 2), shape, mode='bilinear').permute(0, 2, 3, 1)
    return semantic_corr

def read_keyframe(target, edit_dir, shape=None):
    items = {
        'img':        read_images([edit_dir/'keyframe_{}.png'.format(target)], shape).to(device),
        'img_crop':   read_images([edit_dir/'keyframe_{}_crop.png'.format(target)]).to(device),
        'mask':         read_images([edit_dir/'keyframe_{}_mask.png'.format(target)], shape).to(device),
        'mask_crop':    read_images([edit_dir/'keyframe_{}_crop_mask.png'.format(target)]).to(device)
    }

    
    return items

def get_keyframe_item(edit_dir, shape, new_crop_area=None, original_crop_area=None):
    src = read_keyframe('input', edit_dir, shape)
    tgt = read_keyframe('edited', edit_dir, shape)
    corr = get_correspondence(edit_dir).to(device)
    items = {
        'src_img':      src['img'],
        'src_img_crop': src['img_crop'],
        'src_mask':     src['mask'],
        'src_mask_crop':src['mask_crop'],
        'tgt_img':      tgt['img'],
        'tgt_img_crop': tgt['img_crop'],
        'tgt_mask':     tgt['mask'],
        'tgt_mask_crop':tgt['mask_crop'],
        'corr':         torch.ones(1, shape[0], shape[1], 2).to(device) * -2,
        'corr_crop':    corr
    }
    if new_crop_area is not None and original_crop_area is not None:
        scale = 1 / (new_crop_area[1] - new_crop_area[0])
        
        items['corr'] = items['corr'].permute(0, 3, 1, 2)
        items['corr_crop'] = rescale_from_crop(items['corr_crop'], original_crop_area).permute(0, 3, 1, 2)
        for name in ['src_img', 'tgt_img', 'src_mask', 'tgt_mask', 'corr']:
            item_resized = F.interpolate(items[name], scale_factor=scale, mode='bilinear')
        
            H, W = item_resized.shape[-2:]
            t = int((H - 1) * original_crop_area[0])
            b = int((H - 1) * original_crop_area[1])
            l = int((W - 1) * original_crop_area[2])
            r = int((W - 1) * original_crop_area[3])
            item_resized[:, :, t:b, l:r] = F.interpolate(items[name + '_crop'], (b-t, r-l), mode='bilinear')

            t = int((H - 1) * new_crop_area[0])
            b = int((H - 1) * new_crop_area[1])
            l = int((W - 1) * new_crop_area[2])
            r = int((W - 1) * new_crop_area[3])
            items[name + '_crop'] = F.interpolate(item_resized[:, :, t:b, l:r], shape, mode='bilinear')
        items['corr'] = items['corr'].permute(0, 2, 3, 1)
        items['corr_crop'] = scale_to_crop(items['corr_crop'].permute(0, 2, 3, 1), new_crop_area)

    return items

def read_training_frames(data_dir, ids, shape):
    img_paths = sorted(list((data_dir/'images').glob('*.png')) + list((data_dir/'images').glob('*.jpg')))
    mask_paths = sorted(list((data_dir/'masks').glob('*.png')))
    imgs = read_images([img_paths[i] for i in ids], shape).to(device)
    masks = read_images([mask_paths[i] for i in ids], shape).to(device)
    return imgs, masks

def write_rgb(path, tensor):
    imageio.imwrite(path, np.uint8(tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255))

def write_mask(path, tensor):
    imageio.imwrite(path, np.uint8(tensor[0][0].detach().cpu().numpy() * 255))

def get_superresolution(img_path):
    os.system(f'python scripts/run_superresolution.py {img_path}')
    return imageio.imread(img_path)
