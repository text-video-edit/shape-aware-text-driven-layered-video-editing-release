import torch
import torch.nn.functional as F
import torchvision
from utils import *
import imageio
import numpy as np
import oflibpytorch as of

class Deformer(torch.nn.Module):
    def __init__(self):
        super(Deformer, self).__init__()
        
    def initialize(self, deform_uv, project_uv=None, mask=None, scale_x=1, scale_y=1):
        '''
        Input:
            deform_uv:  [N, H, W, 2]
            project_uv: [N, H', W', 2]
            mask:       [N, 1, H, W]
        Return:
        '''
        N, H, W, _ = deform_uv.shape
        xy = get_meshgrid(H, W, normalize=True).to(deform_uv.device)
        
        valid_mask = get_uv_mask(deform_uv)
        
        self.delta = (deform_uv.permute(0, 3, 1, 2) - xy) * valid_mask
        self.coord = xy
        self.mask = mask if mask is not None else valid_mask
        
        
        if project_uv is not None:
            self.delta = backward_warp(self.delta, project_uv)
            self.coord = backward_warp(self.coord, project_uv)
            self.mask = mask if mask is not None else backward_warp(valid_mask, project_uv)

    def update_delta(self, delta):
        self.delta = delta

    def forward(self, uv, scale_x=1, scale_y=1):
        N, H, W, _ = uv.shape
        
        delta = self.delta

        delta_ = backward_warp(delta.repeat(N, 1, 1, 1),      uv)
        coord_ = backward_warp(self.coord.repeat(N, 1, 1, 1), uv)
        mask_  = backward_warp(self.mask.repeat(N, 1, 1, 1),  uv)
        mask_ = (mask_ > 0.5).float()
        
        
        dx = 2 / (W - 1) * 1
        dy = 2 / (H - 1) * 1
        coord_x = coord_.clone()
        coord_x[:, 0] += dx 
        coord_y = coord_.clone()
        coord_y[:, 1] += dy
        
        coords_, coords_x, coords_y = [], [], []
        for i in range(N):
            tps_map = get_tps_mapping(coord_.permute(0, 2, 3, 1)[i:i+1], mask=mask_[i:i+1], inverse=True, lambd=100)
        
            coords_.append(tps_map(coord_[i:i+1]))
            coords_x.append(tps_map(coord_x[i:i+1]))
            coords_y.append(tps_map(coord_y[i:i+1]))
        
        coord_ = torch.cat(coords_, 0)
        coord_x = torch.cat(coords_x, 0)
        coord_y = torch.cat(coords_y, 0)

        a = (coord_x[:, 0] - coord_[:, 0]) / dx 
        b = (coord_x[:, 1] - coord_[:, 1]) / dy
        c = (coord_y[:, 0] - coord_[:, 0]) / dx
        d = (coord_y[:, 1] - coord_[:, 1]) / dy

        local_trans_mat = torch.stack((a, c, b, d), -1).view(N, H, W, 2, 2).view(N, -1, 2, 2)

        delta_ = delta_.permute(0, 2, 3, 1).view(N, -1, 2, 1)
        delta_ = local_trans_mat @ delta_
        delta_ = delta_.view(N, H, W, 2)
        
        grid = get_meshgrid(H, W, normalize=True).to(uv.device)
        deform = grid.permute(0, 2, 3, 1) + delta_
        
        m = mask_.permute(0, 2, 3, 1)
        deform = deform * m + torch.ones_like(deform).to(deform.device) * -2 * (1 - m)

        return deform, mask_
