import math

import torch
import torch.nn as nn
import torchvision

from kornia.geometry.transform.imgwarp import warp_perspective

import utils.geom
import utils.vox
import utils.basic

from models.encoder import freeze_bn, UpsamplingConcat


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, feat2d=128):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(backbone)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.feat2d = feat2d
        self.head_conv = 128

        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        self.up1_skip = UpsamplingConcat(256 + in_channels, in_channels)

        # bev
        self.bev_heads = nn.ModuleDict()
        bev_head_config = {
            'center': n_classes,
            'offset': 4,
            'size': 3,
            'rot': 8,
            # 'id_feat': self.reid_feat,
        }
        for name, out_channels in bev_head_config.items():
            self.bev_heads[name] = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(self.head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, out_channels, kernel_size=1, padding=0),
            )
            if name == 'center':
                self.bev_heads[name][-1].bias.data.fill_(-2.19)

        # img
        self.img_heads = nn.ModuleDict()
        self.img_heads_config = {
            'center': n_classes,
            # 'offset': 2,
            # 'size': 2,
            # 'id_feat': self.reid_feat,
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

    def forward(self, x, feat_cams, bev_flip_indices=None):
        b, c, h, w = x.shape
        x_raw = x

        # pad input
        m = 16
        ph, pw = math.ceil(h / m) * m - h, math.ceil(w / m) * m - w
        pt, pb = ph // 2, ph - (ph // 2)
        pl, pr = pw // 2, pw - (pw // 2)
        x = torch.nn.functional.pad(x, [pl, pr, pt, pb])

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        # Unpad
        x = x[..., pt:pt + h, pl:pl + w]

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        # bev
        out_bev = {'bev_raw': x_raw, 'bev_feat': x}
        for name, head in self.bev_heads.items():
            out_bev[f'instance_{name}'] = head(x)

        # img
        out_img = {'img_raw_feat': feat_cams}
        for name, head in self.img_heads.items():
            out_img[f'img_{name}'] = head(feat_cams)

        return {**out_bev, **out_img}
    
    def align_output(self, out_dict, pix_T_cams, cams_T_global, ref_T_global, rgb_cams_shape):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        """
        
        B, S, C, H, W = rgb_cams_shape
        
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4
        
        img_center = out_dict['img_center']
        
        bev_center = out_dict['instance_center']
        bev_center_ = bev_center.repeat(S, 1, 1, 1)
        
        _ , _, Hi, Wi = img_center.shape
        
        sy = Hi / float(H)
        sx = Wi / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4
        
        global_T_cams_ = torch.inverse(cams_T_global_)
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)
        cams_T_ref_ = torch.inverse(ref_T_cams_)
        
        
        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3
        ref_T_featpix_ = torch.inverse(featpix_T_ref_)
        
        
        projected_bev_center = warp_perspective(bev_center_, ref_T_cams_, (Hi, Wi), align_corners=False)
        out_dict['projected_bev_center'] = projected_bev_center
        
        # projected_img_center = warp_perspective(img_center, ref_T_featpix_, (Hi, Wi), align_corners=False)
        # out_dict['projected_img_center'] = projected_img_center
        
        return out_dict