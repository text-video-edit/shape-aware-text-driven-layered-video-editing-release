import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler
from models.backbone.skip import skip


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def define_G(cfg, in_channel, out_channel, decorr_rgb=False, sigmoid=True, tanh=False):
    netG = skip(
        in_channel,
        out_channel,
        num_channels_down=[cfg["skip_n33d"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n33d"], int)
        else cfg["skip_n33d"],
        num_channels_up=[cfg["skip_n33u"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n33u"], int)
        else cfg["skip_n33u"],
        num_channels_skip=[cfg["skip_n11"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n11"], int)
        else cfg["skip_n11"],
        need_bias=True,
        need_sigmoid=sigmoid,
        need_tanh=tanh,
        decorr_rgb=decorr_rgb,
    )
    return netG

class PointsMLP(nn.Module):
    def __init__(self, num_points):
        super(PointsMLP, self).__init__()
        self.fc0 = nn.Linear(num_points * 2, 320)
        self.fc1 = nn.Linear(320, 320)
        self.fc2 = nn.Linear(320, num_points * 2)
        self.num_points = num_points
    def forward(self, x):
        '''
        Input:
            x: [1, num_points, 2]
        '''
        x = x.contiguous().view(-1, self.num_points * 2)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# https://github.com/WarBean/tps_stn_pytorch/blob/master/mnist_model.py
class BoundedGridLocNet(nn.Module):

    def __init__(self, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        num_points = target_control_points.shape[1]
        self.mlp = PointsMLP(num_points)

        bias = torch.from_numpy(np.arctanh(target_control_points.detach().cpu().numpy()))
        bias = bias.reshape(-1)
        self.mlp.fc2.bias.data.copy_(bias)
        self.mlp.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.mlp(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        num_points = target_control_points.shape[1]
        self.mlp = PointsMLP(num_points)

        bias = target_control_points.view(-1)
        self.mlp.fc2.bias.data.copy_(bias)
        self.mlp.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.mlp(x)
        return points.view(batch_size, -1, 2)
