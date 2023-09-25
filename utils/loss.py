# some codes from https://github.com/omerbt/Text2LIVE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torchvision
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loss(torch.nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()
        self.vgg_loss = VGGLoss().to(device)

        self.opt = opt

    def resize(self, img, scale=None, shape=None):
        if scale is None and shape is not None:
            return F.interpolate(img, shape, mode='bilinear')
        else:
            return F.interpolate(img, scale_factor=scale, mode='bilinear')

    def pixel_loss(self, img1, img2, mask=None, scales=[0, 1, 2]):
        
        loss = 0
        for s in scales:
            s = 2 ** (-s)
            if mask is None:
                loss += F.l1_loss(self.resize(img1, s), self.resize(img2, s))
            else:
                m = self.resize(mask, s).detach()
                loss += F.l1_loss(self.resize(img1, s) * m, self.resize(img2, s) * m, reduction='sum') / m.sum()
        return loss

    def smooth_loss(self, _map, scales=[0]):
        loss = 0
        for s in scales:
            s = 2 ** (-s)
            _map_scaled = self.resize(_map, s)
            grad_x = torch.abs(_map_scaled[:, :, :, :-1] - _map_scaled[:, :, :, 1:])
            grad_y = torch.abs(_map_scaled[:, :, :-1, :] - _map_scaled[:, :, 1:, :])
            loss += grad_x.mean() + grad_y.mean()
        return loss
   
    def edge_aware_smooth_loss(self, _map, _ref, scales=[0]):
        loss = 0
        for s in scales:
            s = 2 ** (-s)
            _map_scaled = self.resize(_map, s)
            _ref_scaled = self.resize(_ref, s)

            grad_x = torch.abs(_map_scaled[:, :, :, :-1] - _map_scaled[:, :, :, 1:])
            grad_y = torch.abs(_map_scaled[:, :, :-1, :] - _map_scaled[:, :, 1:, :])

            ref_grad_x = torch.mean(torch.abs(_ref_scaled[:, :, :, :-1] - _ref_scaled[:, :, :, 1:]), 1, keepdim=True)
            ref_grad_y = torch.mean(torch.abs(_ref_scaled[:, :, :-1, :] - _ref_scaled[:, :, 1:, :]), 1, keepdim=True)

            grad_x *= torch.exp(-ref_grad_x)
            grad_y *= torch.exp(-ref_grad_y)
            loss += grad_x.mean() + grad_y.mean()

        return loss

    def forward(self, items):
        loss = 0.
        loss_target_pixel = 0
        loss_target_perceptual = 0
        loss_delta_smooth = 0
        loss_texture_smooth = 0
        loss_texture_pixel = 0
        loss_mask_smooth = 0
        loss_delta_pixel= 0
        loss_corr_smooth = 0
        loss_corr_mask = 0

        if items['rec_target'] is not None:
            if self.opt.lambda_target_pixel > 0:
                loss_target_pixel = self.pixel_loss(items['rec_target'], items['target']) * self.opt.lambda_target_pixel

            if self.opt.lambda_target_perceptual > 0:
                loss_target_perceptual = self.vgg_loss(items['rec_target'], items['target']) * self.opt.lambda_target_perceptual
        '''
        if self.opt.lambda_delta_smooth > 0:
            loss_delta_smooth = self.smooth_loss(items['delta']) * self.opt.lambda_delta_smooth
        '''
        if self.opt.lambda_texture_smooth > 0:
            loss_texture_smooth = self.smooth_loss(items['texture'], self.opt.texture_smooth_scales) * self.opt.lambda_texture_smooth
        '''
        if self.opt.lambda_texture_pixel > 0:
            loss_texture_pixel = self.pixel_loss(items['texture'], items['initial_texture'], items['initial_mask']) * self.opt.lambda_texture_pixel
        
        if self.opt.lambda_delta_pixel > 0:
            loss_delta_pixel = self.pixel_loss(items['delta'], items['initial_delta'], items['initial_mask']) * self.opt.lambda_delta_pixel
      
        if self.opt.lambda_mask_smooth > 0:
            loss_mask_smooth = self.smooth_loss(items['texture_mask']) * self.opt.lambda_mask_smooth

        if 'corr' in items.keys() and self.opt.lambda_corr_smooth > 0:
            loss_corr_smooth = self.smooth_loss(items['corr']) * self.opt.lambda_corr_smooth
        '''
        if 'tgt_warp_mask' in items.keys() and self.opt.lambda_corr_mask > 0:
            loss_corr_mask = self.pixel_loss(items['tgt_warp_mask'], items['src_mask']) * self.opt.lambda_corr_mask
        

        loss =  loss_target_pixel + \
                loss_target_perceptual + \
                loss_delta_smooth + \
                loss_texture_smooth + \
                loss_mask_smooth + \
                loss_texture_pixel + \
                loss_delta_pixel + \
                loss_corr_smooth + \
                loss_corr_mask

        return loss

class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.
    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).
    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.
    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)
