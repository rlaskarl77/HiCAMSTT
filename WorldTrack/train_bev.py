import os.path as osp
import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import kornia
import cv2

from models import BEV
from models.loss import FocalLoss, compute_rot_loss
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics


class BEVModel(pl.LightningModule):
    def __init__(
            self,
            model_name='bev',
            encoder_name='res18',
            learning_rate=0.001,
            resolution=(200, 4, 200),
            bounds=(-75, 75, -75, 75, -1, 5),
            num_cameras=None,
            depth=(100, 2.0, 25),
            scene_centroid=(0.0, 0.0, 0.0),
            max_detections=60,
            conf_threshold=0.25,
            num_classes=1,
            use_temporal_cache=True,
            z_sign=1,
            feat2d_dim=128,
    ):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth
        self.conf_threshold = conf_threshold

        # Loss
        self.center_loss_fn = FocalLoss()

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras
        if model_name == 'bev':
            self.model = BEV(num_cameras=num_cameras, feat2d_dim=feat2d_dim, 
                             encoder_type=self.encoder_name, num_classes=num_classes)
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.save_hyperparameters()

    def forward(self, item):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        output = self.model(
            rgb_cams=item['img'],
        )

        return output


    def loss(self, target, output):

        center_img_e = output['img_center']
        offset_img_e = output['img_offset']
        size_img_e = output['img_size']
        depth_img_e = output['img_depth']

        B, S = target['center_img'].shape[:2]
        
        valid_img = basic.pack_seqdim(target['valid_img'], B)
        center_img_g = basic.pack_seqdim(target['center_img'], B)
        offset_img_g = basic.pack_seqdim(target['offset_img'], B)
        size_img_g = basic.pack_seqdim(target['size_img'], B)
        depth_img_g = basic.pack_seqdim(target['depth_img'], B)

        # img loss
        center_img_loss = self.center_loss_fn(basic.sigmoid(center_img_e), center_img_g) / S
        size_img_loss = torch.abs(size_img_g - size_img_e).sum(dim=1, keepdim=True)
        size_img_loss = basic.reduce_masked_mean(size_img_loss, valid_img)
        offset_img_loss = torch.abs(offset_img_g - offset_img_e).sum(dim=1, keepdim=True)
        offset_img_loss = basic.reduce_masked_mean(offset_img_loss, valid_img)
        depth_img_loss = torch.abs(torch.log(depth_img_g) - depth_img_e).sum(dim=1, keepdim=True)
        depth_img_loss = basic.reduce_masked_mean(depth_img_loss, valid_img)

        loss_dict = {
            'center_loss': 10 * center_img_loss,
            'offset_loss': offset_img_loss,
            'size_loss': size_img_loss,
            'depth_loss': center_img_loss,
        }
        
        total_loss = sum(loss_dict.values())

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)
        
        # if batch_idx < 5:
        #     self.plot_data_train(item, target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        # if batch_idx % 100 == 1:
        #     self.plot_data(target, output, batch_idx)
        # self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        self.log('val_depth', loss_dict['depth_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss

    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # save plots to tensorboard in eval loop
        writer = self.logger.experiment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax2.imshow(center_e[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        
    
    def plot_data_train(self, item, target, output, batch_idx=0):
        
        # save plots to tensorboard in training loop
        writer = self.logger.experiment
        
        imgs = item['img'] # B, S, 3, H, W
        imgs0 = imgs[0]
        imgs0 = imgs0.permute(0, 2, 3, 1).cpu().numpy()
        
        S = imgs0.shape[0]
        fig, axs = plt.subplots(1, S, figsize=(S*4, 4))
        for i, ax in enumerate(axs):
            ax.imshow(imgs0[i])
            ax.axis('off')
            ax.set_title(f'cam{i+1}')
            
        plt.tight_layout()
        writer.add_figure(f'plot/input{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)
        
        center_e = output['instance_center'].detach()[-1].amax(0).sigmoid().squeeze().cpu().numpy()
        center_g = target['center_bev'].detach()[-1].amax(0).sigmoid().squeeze().cpu().numpy()

        # save plots to tensorboard in eval loop
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g)
        ax2.imshow(center_e)
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(f'plot/train{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)


if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")


    cli = MyLightningCLI(BEVModel)
