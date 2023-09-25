import yaml
import torch
from .networks import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AtlasModel(torch.nn.Module):
    def __init__(self, alpha, cfg='models/config.yaml'):
        super(AtlasModel, self).__init__()
        self.cfg = yaml.safe_load(open(cfg))
        self.appear_net = define_G(self.cfg, 3, 4, decorr_rgb=True, sigmoid=True, tanh=False).to(device)
        self.deform_net = define_G(self.cfg, 2, 2, sigmoid=False, tanh=True).to(device)
        
        for n, p in self.deform_net.named_parameters():
            if n == '8.1.weight' or n == '8.1.bias':
                p.data.zero_()
        
        self.fixed_alpha = alpha.detach().requires_grad_()

    def forward(self, inputs):
        outputs = {}
         
        rgba = self.appear_net(inputs['texture'])
        delta = self.deform_net(inputs['delta'])

        rgb = rgba[:, :3]
        alpha = rgba[:, 3:]
        outputs['alpha'] = 1 - self.fixed_alpha #alpha
        outputs['texture'] = rgb #* alpha + inputs['texture'] * (1 - alpha)
        outputs['delta'] = delta #* alpha + inputs['delta'] * (1 - alpha)

        return outputs

class TPSModel(torch.nn.Module):
    def __init__(self, tps_points, bounded=True):
        super(TPSModel, self).__init__()
        if bounded:
            self.corr_net = BoundedGridLocNet(tps_points).to(device)
        else:
            self.corr_net = UnBoundedGridLocNet(tps_points).to(device)
    def forward(self, x):
        return self.corr_net(x)
