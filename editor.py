import sys, os, argparse, time, json, yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
from path import Path
from tqdm import tqdm

from deformer import * 
from utils import *
import imageio
import imageio.v3 as iioi

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from models.model import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Editor(torch.nn.Module):
    def __init__(self, opt):
        super(Editor, self).__init__()
        
        edit_dir = Path(opt.edit_dir)
        data_dir = Path(os.path.split(edit_dir)[0])
        nla_dir = data_dir/'nla_outputs'

        edit_meta = yaml.safe_load(open(edit_dir/'edit_meta.yaml'))
        k = int(edit_meta['keyframe_index'])
        self.text_prompt = edit_meta['prompt']

        # load atlases and UV, alpha maps from NLA dir
        H, W = opt.height, opt.width
        
        maps = torch.load(nla_dir/'maps')
        self.textures = {
            'src1': read_images([nla_dir/'texture1.png']).to(device),
            'src2': read_images([nla_dir/'texture2.png']).to(device),
            'tgt2': read_images([opt.edit_bg]).to(device) if opt.edit_bg is not None else None,
        }

        self.maps = {
            'uv1': resize_uv(maps['uv1'].to(device), (H, W)),
            'uv2': resize_uv(maps['uv2'].to(device), (H, W)),
            'uv1_crop': resize_uv(maps['uv1_crop'].to(device), (H, W)),
            'uv2_crop': resize_uv(maps['uv2_crop'].to(device), (H, W)),
            'alpha': F.interpolate(maps['alpha'].unsqueeze(1).to(device), (H, W), mode='bilinear'),
            'alpha_crop': F.interpolate(maps['alpha_crop'].unsqueeze(1).to(device), (H, W), mode='bilinear'),
        }
        # load target edit frame 
        N = self.maps['uv1'].shape[0]
        
        self.H = H
        self.W = W
        self.N = N
        self.k = k
        self.data_dir = data_dir
        self.edit_dir = edit_dir
        self.edit_meta = edit_meta

        # keyframe
        texture_size = self.textures['src1'].shape[-2:]
        kf_crop_area = maps['crop_areas1'][k].tolist()
        kf = get_keyframe_item(edit_dir, (H, W), kf_crop_area, edit_meta['crop_area'])
       
        uv_k = self.maps['uv1_crop'][[k]]
        uv_inv_k = tps(uv_k, inverse=True, shape=texture_size)
        
        # transform crop area in texture
        corr = tps(kf['corr_crop'], mask=get_uv_mask(kf['corr_crop']), 
                   nx=20, ny=20, lambd=0.0, return_full=True)
        
        self.df = Deformer().to(device)
        self.kf = kf
        self.corr = corr
        self.uv_inv_k = uv_inv_k
        
        self.df.initialize(self.corr['warp'], uv_inv_k)
        self.textures['tgt1'] = backward_warp(backward_warp(kf['tgt_img_crop'], corr['warp']), uv_inv_k)
        self.textures['tgt_mask1'] = backward_warp(backward_warp(kf['tgt_mask_crop'], corr['warp']), uv_inv_k)
        self.textures['tgt_mask1'] = self.textures['tgt_mask1'].detach().float().requires_grad_(True)
        self.textures['tgt_delta1'] = self.df.delta.detach().float().requires_grad_(True)

        self.initials = {
            'texture': self.textures['tgt1'],
            'delta': self.df.delta,
            'corr_tps': corr['tgt_pts'],
        }
        
        self.model_atlas = AtlasModel(self.textures['tgt_mask1']).to(device)
        self.model_corr = TPSModel(corr['tgt_pts']).to(device) if 'corr_tps' in self.initials.keys() else None
        
        # rendering utilities
        sigma1, sigma2 = 5, 10
        self.blur_func1 = GaussianBlur((sigma1 * 4 + 1, sigma1 * 4 + 1), sigma=sigma1)
        self.blur_func2 = GaussianBlur((sigma2 * 4 + 1, sigma2 * 4 + 1), sigma=sigma2)
        pad_w = sigma2 * 2
        self.featherize = lambda X: self.blur_func2(F.pad(X, (pad_w, pad_w, pad_w, pad_w)))[:, :, pad_w:-pad_w, pad_w:-pad_w]

    def forward_model(self):
        if 'corr_tps' in self.initials.keys():
            out_tgt = self.model_corr(self.initials['corr_tps'])
            corr = compute_tps_grid(self.corr['src_pts'], out_tgt, self.corr['grid'], lambd=0)
            corr = corr.view(1, self.H, self.W, 2)
        
        self.df.initialize(corr, self.uv_inv_k)
        self.initials['delta'] = self.df.delta
        
        outputs = self.model_atlas(self.initials)
        outputs['corr'] = corr
        return outputs
    
    def initialize_render(self, forward_model=False, train=False):
        if train or forward_model:
            outputs = self.forward_model()
            corr = outputs['corr']
            texture = outputs['texture']
            delta = self.df.delta + outputs['delta']
            self.df.update_delta(delta)
        else:
            corr = self.corr['warp']
            texture = self.textures['tgt1']
            self.df.initialize(corr, self.uv_inv_k)
        
        texture_valid = backward_warp(backward_warp(
            torch.ones_like(self.kf['tgt_img_crop']).to(device), corr), self.uv_inv_k)
        
        texture = texture * self.featherize(texture_valid)
        texture_mask = ((self.textures['tgt_mask1'] + texture_valid[:, :1]) > 0.5).float()
        
        if not train:
            return texture, texture_mask
        else:
            return texture, texture_mask, delta, corr

    def render(self, index, crop=False, crop_detail=False):
        
        texture1, texture_mask, delta, corr = self.initialize_render(train=True)
        tgt_warp_mask = backward_warp(self.kf['tgt_mask_crop'], corr)

        B = len(index) 
        
        if not crop:
            uv1 = self.maps['uv1'][index]
            uv2 = self.maps['uv2'][index]
            alpha = self.maps['alpha'][index]
        else:
            uv1 = self.maps['uv1_crop'][index]
            uv2 = self.maps['uv2_crop'][index]
            alpha = self.maps['alpha_crop'][index]

        deform, mask = self.df(uv1)
        
        deform_inv = []
        for i in range(B):
            deform_inv.append(tps(deform[[i]], inverse=True, mask=mask[[i]], lambd=0.05))
        deform_inv = torch.cat(deform_inv, 0)
        
        img = self.composite(
                    texture1, 
                    self.textures['src2'], 
                    texture_mask, 
                    uv1, 
                    uv2, 
                    deform_inv,
                    alpha,
                    self.textures['src1'],
                    mask, 
                    )
        img_detail = None
        if crop_detail:
            half_w = 0.15 + torch.rand(1) * (0.5 - 0.15)
            cx, cy = torch.rand(2) * (1 - 2 * half_w) + half_w
            
            l = int((cx - half_w) * (self.W - 1))
            r = int((cx + half_w) * (self.W - 1))
            t = int((cy - half_w) * (self.H - 1))
            b = int((cy + half_w) * (self.H - 1))
            
            img_detail = F.interpolate(img[:, :, t:b, l:r], (self.H, self.W), mode='bilinear')

        return {
            'image': img,
            'image_detail': img_detail,
            'image_mask': mask,
            'target': self.kf['tgt_img_crop'] if crop else self.kf['tgt_img'],
            'texture': texture1,
            'delta': delta,
            'texture_mask': texture_mask,
            'initial_texture': self.textures['tgt1'],
            'initial_mask': self.textures['tgt_mask1'],
            'initial_delta': self.textures['tgt_delta1'],
            'corr': corr,
            'tgt_warp_mask': tgt_warp_mask,
            'src_mask': self.kf['src_mask_crop'],
        }

    def composite(self, texture1, texture2, texture_mask, uv1, uv2, deform_inv, alpha, texture1_org=None, mask_org=None):
        deformed_uv1 = warp_uv(uv1, deform_inv)
        N = deformed_uv1.shape[0]
        mask = backward_warp(texture_mask.repeat(N, 1, 1, 1), deformed_uv1)
        
        deform_smooth_mask = self.featherize((self.blur_func2(backward_warp(torch.ones_like(alpha).to(device), deform_inv)) > 0.9).float())
        deform_alpha = backward_warp(alpha, deform_inv, padding='border') * deform_smooth_mask 
        
        img1 = backward_warp(texture1.repeat(N, 1, 1, 1), deformed_uv1) * deform_alpha
        
        if False: #and texture1_org is not None:
            valid = (mask + mask_org).clamp(0, 1)
            valid = (self.blur_func2(valid) > 0.05).float()
            valid = self.blur_func2(valid)
            img1_org = backward_warp(texture1_org.repeat(N, 1, 1, 1), uv1) * alpha
            img1 = img1 * valid + img1_org * (1 - valid)
            blend_alpha = deform_alpha * valid + alpha * (1 - valid)
        else:
            blend_alpha = deform_alpha
        
        img2 = backward_warp(texture2.repeat(N, 1, 1, 1), uv2)
        img = img1 * blend_alpha + img2 * (1 - blend_alpha)
        return img

    @torch.no_grad()
    def render_video(self, output_path, smoothness=5, edit_bg=False, interpolate_fg=False, interpolate_bg=False): 
        texture1, texture_mask = self.initialize_render(forward_model=True, train=False)

        texture1_path = self.edit_dir/'texture1.png'
        write_rgb(texture1_path, texture1)
        texture1 = get_superresolution(texture1_path)
        texture1 = torch.FloatTensor(texture1).permute(2, 0, 1).unsqueeze(0).to(device) / 255.
        texture2 = self.textures['tgt2'] if self.textures['tgt2'] is not None and edit_bg else self.textures['src2']

        original_masks = []
        video_writer = imageio.get_writer(output_path, fps=10)
       
        temporal_tps = TemporalTPS(self.H, self.W, nx=20, ny=20, lambd=0.05)

        if 'clip' in self.edit_meta.keys():
            iteration = range(int(self.edit_meta['start_index']), int(self.edit_meta['end_index']))
        else:
            iteration = range(self.N)

        if interpolate_fg:
            self.initialize_deform_interpolation()
            interpolation1 = self.initialize_texture_interpolation(self.textures['src1'], texture1)
        if interpolate_bg:
            interpolation2 = self.initialize_texture_interpolation(self.textures['src2'], texture2)

        for i in tqdm(range(self.N)):
            
            if interpolate_fg:
                texture1 = self.interpolate_deform(i, 25, 58)

            uv1 = self.maps['uv1'][[i]]
            deform, mask = self.df(uv1)
            original_masks.append(mask)
            temporal_tps.append(deform, mask=mask)

        temporal_tps.smooth(smoothness)
         
        for i in tqdm(iteration):
            if interpolate_fg:
                texture1 = self.interpolate_texture(interpolation1, i, 25, 58)
            if interpolate_bg:
                texture2 = self.interpolate_texture(interpolation2, i, 25, 58)
            img = self.composite(
                    texture1,
                    texture2,
                    texture_mask, 
                    self.maps['uv1'][[i]], 
                    self.maps['uv2'][[i]], 
                    temporal_tps.get(i, inverse=True),
                    self.maps['alpha'][[i]],
                    self.textures['src1'],
                    original_masks[i], 
                    )

            video_writer.append_data(np.uint8(img[0].permute(1, 2, 0).detach().cpu().numpy() * 255))

        video_writer.close()
  
    def initialize_deform_interpolation(self):
        self.delta_saved = self.df.delta

    def interpolate_deform(self, i, start, end):
        inter_w = np.clip((i - start) / (end - start), 0, 1)
        
        delta = self.delta_saved * inter_w
        self.df.delta = delta

    def initialize_texture_interpolation(self, begin_texture, end_texture):
        interpolation_item = {
            'begin': begin_texture,
            'end': F.interpolate(end_texture, begin_texture.shape[-2:], mode='bilinear')
        }
        return interpolation_item

    def interpolate_texture(self, item, i, start, end):
        inter_w = np.clip((i - start) / (end - start), 0, 1)
        
        texture = item['begin'] * (1 - inter_w) + item['end'] * inter_w
        return texture
        


    def original_render_video(self, output_path, edit_bg=False):
        
        video_writer = imageio.get_writer(output_path, fps=10)
        if 'clip' in self.edit_meta.keys():
            iteration = range(int(self.edit_meta['start_index']), int(self.edit_meta['end_index']))
        else:
            iteration = range(self.N)
        
        for i in tqdm(iteration):
            uv1 = self.maps['uv1'][[i]]
            uv2 = self.maps['uv2'][[i]]
            texture1 = self.textures['src1']
            texture2 = self.textures['tgt2'] if self.textures['tgt2'] is not None and edit_bg else self.textures['src2']
            alpha = self.maps['alpha'][[i]]
            
            img1 = backward_warp(texture1, uv1)
            img2 = backward_warp(texture2, uv2)
            img = img1 * alpha + img2 * (1 - alpha)

            video_writer.append_data(np.uint8(img[0].permute(1, 2, 0).detach().cpu().numpy() * 255))
        video_writer.close()

    def get_crops(self, indices):
        crop_grids = []
        grid = get_meshgrid(self.H, self.W, normalize=True).permute(0, 2, 3, 1)
        for i in indices:
            t, b, l, r = self.crop_areas[i] 
            crop_grid = rescale_from_zoom(grid.clone(), (t, b, l, r))
            crop_grids.append(crop_grid)
        return torch.cat(crop_grids, 0).to(device)

    def apply_crop(self, images, indices):
        crop_grids = self.get_crops(indices)
        crop_images = backward_warp(images, crop_grids)
        return crop_images

    def get_params(self, lr):
        params = [
            {'params': self.model_atlas.parameters(), 'lr': lr},
            {'params': self.model_corr.parameters(), 'lr': lr},
        ]
        return params

