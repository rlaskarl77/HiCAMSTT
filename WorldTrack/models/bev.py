import math
import torch
import torch.nn as nn
import torchvision

import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, \
    Encoder_eff, Encoder_swin_t, Encoder_res34, freeze_bn, UpsamplingConcat

class BEV(nn.Module):
    def __init__(self,
                 num_cameras=None,
                 num_classes=None,
                 feat2d_dim=128,
                 encoder_type='res18',
                 device=torch.device('cuda')):
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])

        self.encoder_type = encoder_type
        self.num_cameras = num_cameras

        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        self.decoder = Decoder(
            n_classes=num_classes,
            feat2d=self.feat2d_dim,
        )

    def forward(self, rgb_cams):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        """
        B, S, C, H, W = rgb_cams.shape
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)  # B*S,3,H,W
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,latent_dim,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape
        out_dict = self.decoder(feat_cams_)
        return out_dict

class Decoder(nn.Module):
    def __init__(self, n_classes, feat2d=128):
        super().__init__()
        self.feat2d = feat2d
        self.head_conv = 128

        # img
        self.img_heads = nn.ModuleDict()
        self.img_heads_config = {
            'center': n_classes,
            'offset': 2,
            'size': 2,
            'depth': 1,
        }
        for name, out_channels in self.img_heads_config.items():
            self.img_heads[name] = nn.Sequential(
                nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.feat2d),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feat2d, out_channels, kernel_size=1, padding=0),
            )
            if name == 'center':
                self.img_heads[name][-1].bias.data.fill_(-2.19)

    def forward(self, feat_cams):
        # img
        out_img = {'img_raw_feat': feat_cams}
        for name, head in self.img_heads.items():
            out_img[f'img_{name}'] = head(feat_cams)

        return {**out_img}