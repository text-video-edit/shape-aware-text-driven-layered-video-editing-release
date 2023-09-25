import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import cv2 as cv
import torch.nn as nn
import torchvision
from torch.autograd import Function, Variable
import itertools

from .mask_utils import *
from .warp_utils import *
from .io_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_control_points(src, tgt, mask, nx, ny):
    t, b, l, r = find_mask_border(mask[0][0])
    
    crop = lambda X: X[:, :, t:b+1, l:r+1]
    src = crop(src)
    tgt = crop(tgt)
    mask = crop(mask)
    
    '''
    h, w = src.shape[2], src.shape[3]
    step_y = int(np.floor(h / ny))
    step_x = int(np.floor(w / nx))
    pool = lambda X: F.avg_pool2d(X, (step_y, step_x), stride=(step_y, step_x), ceil_mode=True)
    #'''
    pool = nn.AdaptiveAvgPool2d((ny, nx))

    shift = (tgt - src) * mask
    src_cp = pool(src)
    shift_cp = pool(shift)
    mask_cp = pool(mask)

    shift_cp = shift_cp / (mask_cp + 1e-15)
    tgt_cp = shift_cp + src_cp

    return src_cp.view(1, 2, -1).permute(0, 2, 1), tgt_cp.view(1, 2, -1).permute(0, 2, 1)

def tps(uv, mask=None, nx=25, ny=15, inverse=False, lambd=0.0, shape=None, return_full=False):
    '''
    Input:
        uv: [1, H, W, 2]
        maks: [1, 1, H, W]
    '''
    h = uv.shape[1]
    w = uv.shape[2]

    grid = get_meshgrid(h, w, normalize=True).to(device)

    if mask is None:
        mask = torch.ones(1, 1, h, w).to(device)

    uv = uv.permute(0, 3, 1, 2)
    grid_cp, uv_cp = get_control_points(grid, uv, mask, nx, ny)

    if inverse:
        src_cp, tgt_cp = grid_cp, uv_cp
    else:
        src_cp, tgt_cp = uv_cp, grid_cp

    warp = compute_tps_grid(src_cp, tgt_cp, grid, lambd).view(1, h, w, 2)

    if shape is not None:
        warp = resize_uv(warp, shape)
    if return_full:
        return {
            'warp': warp,
            'src_pts': src_cp,
            'tgt_pts': tgt_cp,
            'grid': grid
        }
    else:
        return warp

def compute_tps_grid(src_cp, tgt_cp, grid, lambd):
    return TPSGridGen(grid[0].view(2, -1).transpose(1, 0), tgt_cp[0], lambd=lambd)(src_cp)

def get_tps_mapping(uv, mask=None, nx=25, ny=15, inverse=False, lambd=0.0):
    '''
    Input:
        uv: [1, H, W, 2]
        maks: [1, 1, H, W]
    '''
    h = uv.shape[1]
    w = uv.shape[2]

    grid = get_meshgrid(h, w, normalize=True).to(device)

    if mask is None:
        mask = get_uv_mask(uv)#torch.ones(1, 1, h, w).to(device)

    uv = uv.permute(0, 3, 1, 2)
    grid_cp, uv_cp = get_control_points(grid, uv, mask, nx, ny)

    if inverse:
        src_cp, tgt_cp = grid_cp, uv_cp
    else:
        src_cp, tgt_cp = uv_cp, grid_cp

    tps_grid = TPSGridGen(grid[0].view(2, -1).transpose(1, 0), tgt_cp[0], lambd, src_cp)
    return tps_grid.map

class TemporalTPS:
    def __init__(self, h, w, nx=24, ny=14, lambd=0.0):
        self.nx = nx
        self.ny = ny
        self.h = h
        self.w = w
        self.lambd = lambd

        self.grid_cp_list = []
        self.uv_cp_list = []

        self.grid = get_meshgrid(self.h, self.w, normalize=True).to(device)

    def append(self, uv, mask=None):
        '''
        Input:
            uv: [1, H, W, 2]
            maks: [1, 1, H, W]
        '''
        if mask is None:
            mask = torch.ones(1, 1, h, w).to(device)

        uv = uv.permute(0, 3, 1, 2)
        grid_cp, uv_cp = get_control_points(self.grid, uv, mask, self.nx, self.ny)
        self.grid_cp_list.append(grid_cp)
        self.uv_cp_list.append(uv_cp)

    def smooth(self, smoothness=5):
        T = len(self.grid_cp_list)
        self.grid_cp_list = torch.cat(self.grid_cp_list, 0)
        self.uv_cp_list = torch.cat(self.uv_cp_list, 0)

        kernel_size = smoothness * 4 + 1
        g_1d_kernel = gaussian_fn(kernel_size, smoothness).to(device)[None, None, :]
        padding = (kernel_size // 2, kernel_size // 2)

        smooth_func = lambda X: F.conv1d(F.pad(X.permute(1, 2, 0).view(-1, 1, T), 
                                               padding, mode='replicate'),
                                         g_1d_kernel).view(-1, 2, T).permute(2, 0, 1)

        self.grid_cp_list = smooth_func(self.grid_cp_list)
        self.uv_cp_list = smooth_func(self.uv_cp_list)

    def get(self, i, inverse=False):
        uv_cp = self.uv_cp_list[i:i+1]
        grid_cp = self.grid_cp_list[i:i+1]
        if inverse:
            src_cp, tgt_cp = grid_cp, uv_cp
        else:
            src_cp, tgt_cp = uv_cp, grid_cp

        warp = compute_tps_grid(src_cp, tgt_cp, self.grid, self.lambd).view(1, self.h, self.w, 2)
        return warp
        

class TPSGridGen(nn.Module):

    def __init__(self, target_coord, target_control_points, lambd=0., source_control_points=None):
        super(TPSGridGen, self).__init__()
        
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()
        self.device = target_control_points.device
        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3).to(self.device)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        target_control_partial_repr += torch.eye(N).to(self.device) * lambd
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_coord.shape[0]
        target_coordinate_partial_repr = compute_partial_repr(target_coord, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1).to(self.device), target_coord
        ], dim = 1)
        
        self.inverse_kernel = inverse_kernel
        self.padding_matrix = torch.zeros(3, 2).to(self.device)
        self.target_coordinate_repr = target_coordinate_repr
        self.target_control_points = target_control_points
        self.HW = HW

        if source_control_points is not None:
            Y = torch.cat([source_control_points, self.padding_matrix.expand(1, 3, 2)], 1)
            self.mapping_matrix = torch.matmul(self.inverse_kernel, Y)

    def map(self, coord):
        '''
        Input:
            coord: [N, 2, H, W]
        Return:
            mapped: [N, 2, H, W]
        '''
        N, _, H, W = coord.shape
        coord_ = coord.view(2, -1).transpose(1, 0)
        coord_partial_repr = compute_partial_repr(coord_, self.target_control_points)
        coord_repr = torch.cat([coord_partial_repr, torch.ones(self.HW, 1).to(self.device), coord_], dim=1)
        mapped = torch.matmul(coord_repr, self.mapping_matrix.repeat(N, 1, 1))
        return mapped.view(N, H, W, 2).permute(0, 3, 1, 2)
        

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


        
# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix
   
