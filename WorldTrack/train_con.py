import os.path as osp
import torch
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
import numpy as np
import kornia
import cv2

from models import Segnet, MVDet, Liftnet, Bevformernet, MVDetr
from models.loss import FocalLoss, compute_rot_loss
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics


class WorldTrackModel(pl.LightningModule):
    def __init__(
            self,
            model_name='segnet',
            encoder_name='res18',
            learning_rate=0.001,
            resolution=(200, 4, 200),
            bounds=(-75, 75, -75, 75, -1, 5),
            num_cameras=None,
            depth=(100, 2.0, 25),
            scene_centroid=(0.0, 0.0, 0.0),
            num_classes=1,
            z_sign=1,
            feat2d_dim=128,
            # decoder
            learn_reid=True,
            reid_feat=128,
            pose_feat=128,
            hard_mask=False,
            temperature=1.,
            cont_type='simclr',
            learn_cont_pose=True,
            temperature_pose=1.,
            pose_cont_thresh=0.5,
            # tracker
            use_reid_tracking=True,
            use_temporal_cache=True,
            max_detections=60,
            conf_threshold=0.25,
            max_cache=32,
            conf_thres=0.1, 
            track_buffer=5,
            lapjv_thresh=0.25,
            lapjv_thresh2=0.5,
            max_spatial_dist=75.,
            max_spatial_dist2=100.,
            dist_alpha=0.5,
            temp_mixing=True,
            lambda_1=1.,
            lambda_2=1.,
            # test mode
            test_dataset_dir: str=None,
            test_mode: str='tracking',
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
        
        # Decoder
        self.learn_reid = learn_reid
        self.decoder_args = {
            learn_reid: learn_reid,
            reid_feat: reid_feat,
            pose_feat: pose_feat,
            hard_mask: hard_mask,
        }
        # contrastive loss
        self.temperature = temperature
        self.cont_type = cont_type # simclr or moco
        self.learn_cont_pose = learn_cont_pose
        self.pose_cont_thresh = pose_cont_thresh
        self.temperature_pose = temperature_pose
        
        if self.cont_type == 'moco':
            self.moco_memory_bank = dict()
        
        # Tracker
        self.tracker_args = {
            use_reid_tracking: use_reid_tracking,
            conf_threshold: conf_threshold,
            conf_thres: conf_thres,
            track_buffer: track_buffer,
            lapjv_thresh: lapjv_thresh,
            lapjv_thresh2: lapjv_thresh2,
            max_spatial_dist: max_spatial_dist,
            max_spatial_dist2: max_spatial_dist2,
            dist_alpha: dist_alpha,
            temp_mixing: temp_mixing,
            lambda_1: lambda_1,
            lambda_2: lambda_2,
        }
        self.test_tracker = JDETracker(**self.tracker_args)

        # Loss
        self.center_loss_fn = FocalLoss()
        
        self.geometric_loss_fn = torch.nn.functional.smooth_l1_loss

        # Temporal cache
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = max_cache
        self.temporal_cache_frames = -2 * torch.ones(self.max_cache, dtype=torch.long) \
            if self.use_temporal_cache else None
        self.temporal_cache = None

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras
        if model_name == 'segnet':
            self.model = Segnet(self.Y, self.Z, self.X, num_cameras=num_cameras, feat2d_dim=feat2d_dim,
                                encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign,
                                decoder_args=self.decoder_args)
        elif model_name == 'liftnet':
            self.model = Liftnet(self.Y, self.Z, self.X, encoder_type=self.encoder_name, feat2d_dim=feat2d_dim,
                                 DMIN=self.DMIN, DMAX=self.DMAX, D=self.D, num_classes=num_classes, z_sign=z_sign,
                                 num_cameras=num_cameras, decoder_args=self.decoder_args)
        elif model_name == 'bevformer':
            self.model = Bevformernet(self.Y, self.Z, self.X, feat2d_dim=feat2d_dim,
                                      encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign,
                                      decoder_args=self.decoder_args)
        elif model_name == 'mvdet':
            self.model = MVDet(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                               num_cameras=num_cameras, num_classes=num_classes, decoder_args=self.decoder_args)
        elif model_name == 'mvdetr':
            self.model = MVDetr(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                                num_cameras=num_cameras, num_classes=num_classes, feat2d_dim=feat2d_dim,
                                decoder_args=self.decoder_args)
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(self.Y, self.Z, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds)
        self.save_hyperparameters()
        
        
        # Test
        self.test_dataset = None if test_dataset_dir is None \
            else 'wildtrack' if 'wildtrack' in test_dataset_dir.lower() \
            else 'multiviewx' if 'multiviewx' in test_dataset_dir.lower() \
            else 'aicity_lt' if 'aicity_lt' in test_dataset_dir.lower() \
            else 'aicity' if 'aicity' in test_dataset_dir.lower() \
            else None
        
        self.test_mode = test_mode
        if self.test_mode == 'tracking':
            self.moda_gt_list, self.moda_pred_list = [], []
            self.mota_gt_list, self.mota_pred_list = [], []
            self.mota_seq_gt_list, self.mota_seq_pred_list = []
        
        elif self.test_mode == 'feature_distance':
            self.max_features = 100
            self.feat_correct_self, self.feat_correct_random = 0, 0
            self.feat_total_self, self.feat_total_random = 0, 0

            self.feat_dist_self = torch.full((self.max_features, self.max_features), -1, device=self.device, dtype=torch.float32)
            self.feat_dist_random = torch.full((self.max_features, self.max_features), -1, device=self.device, dtype=torch.float32)

            self.feat_dist_count_self = torch.zeros(self.max_features, self.max_features, device=self.device)
            self.feat_dist_count_random = torch.zeros(self.max_features, self.max_features, device=self.device)

            self.feat_dist_self_list = []
            self.feat_dist_random_list = []

            self.feat_correct = 0
            self.feat_total = 0
            self.feat_dist = torch.full((self.max_features, self.max_features), 2, device=self.device, dtype=torch.float32)
            self.feat_dist_count = torch.zeros(self.max_features, self.max_features, device=self.device)

            self.feat_dist_list = []
        
        elif self.test_mode == 'pose':
            self.pose_gt_list = []
            self.pose_pred_list = []
        
        elif self.test_mode == 'training' or self.test_mode == 'prediction':
            pass
        
        elif self.test_mode == 'save_features':
            pass
        
        else:
            raise ValueError(f'Unknown test mode {self.test_mode}')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        
    
    def forward(self, item, is_train=True):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        prev_bev = self.load_cache(item['frame'].cpu(), prev=True)

        output = self.model(
            rgb_cams=item['img'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=prev_bev,
        )

        if self.use_temporal_cache:
            self.store_cache(item['frame'].cpu(), 
                            output['bev_raw'].clone().detach())
        
        return output
    
    
    def compute_bev(self, item):
        
        output_prev = self.model(
            rgb_cams=item['img_prev'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=None,
        )
        
        bev_prev_raw = output_prev['bev_raw'].clone().detach()

        output = self.model(
            rgb_cams=item['img_prev'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=bev_prev_raw,
        )
        
        outout_random_prev = self.model(
            rgb_cams=item['img_prev_rand'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=None,
        )
        bev_prev_random_raw = outout_random_prev['bev_raw'].clone().detach()
        
        output_random = self.model(
            rgb_cams=item['img_rand'],
            pix_T_cams=item['intrinsic'],
            cams_T_global=item['extrinsic'],
            ref_T_global=item['ref_T_global'],
            vox_util=self.vox_util,
            prev_bev=bev_prev_random_raw,
        )
        
        return output, output_random
        

    def load_cache(self, frames, prev=False):
        idx = []
        for frame in frames:
            if prev:
                i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            else:
                i = (frame == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            if i.nelement() == 1:
                idx.append(i.item())
        if len(idx) != len(frames):
            return (None, None)
        else:
            return self.temporal_cache[idx]


    def store_cache(self, frames, bev_raw):
        if self.temporal_cache is None:
            dtype = bev_raw.dtype
            device = bev_raw.device
            shape_raw = list(bev_raw.shape)
            shape_raw[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape_raw, device=device, dtype=dtype)
        
        for frame, feat_raw in zip(frames, bev_raw):
            i = (frame == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            # Choose unfilled cache slot
            if i.nelement() == 0:
                i = (self.temporal_cache_frames == -2).nonzero(as_tuple=True)[0]
            # Choose random cache slot
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))

            self.temporal_cache[i[0]] = feat_raw
            self.temporal_cache_frames[i[0]] = frame
        

    def load_identities(self, frame) -> torch.Tensor:
        if frame in self.moco_memory_bank.keys():
            return self.moco_memory_bank[frame]
        return None


    def store_identities(self, target, output):
        frames = target['frame'].cpu()
        bev_feat = output['instance_id_feat']
        id_feat, _, pid_counts = self.get_feature_vec_with_pid(bev_feat, target['pid_bev'])
        
        for i, frame in enumerate(frames):
            prev_identities = self.load_identities(frame)
            feat: torch.Tensor = id_feat[pid_counts[:i].sum():pid_counts[:i+1].sum()]
            if prev_identities is None:
                self.moco_memory_bank[frame] = feat.cpu()
            else:
                assert feat.shape == prev_identities.shape, 'feature dim should be the same'
                self.moco_memory_bank[frame] = prev_identities * 0.9 + feat.cpu() * 0.1
            

    def loss(self, target, output):
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        center_img_e = output['img_center']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)

        center_loss = self.center_loss_fn(basic.sigmoid(center_e), center_g)
        offset_loss = torch.abs(offset_e[:, :2] - offset_g[:, :2]).sum(dim=1, keepdim=True)
        offset_loss = basic.reduce_masked_mean(offset_loss, valid_g)
        tracking_loss = torch.nn.functional.smooth_l1_loss(
            offset_e[:, 2:], offset_g[:, 2:], reduction='none').sum(dim=1, keepdim=True)
        tracking_loss = basic.reduce_masked_mean(tracking_loss, valid_g)
        

        if 'size_bev' in target:
            size_g = target['size_bev']
            rotbin_g = target['rotbin_bev']
            rotres_g = target['rotres_bev']
            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = basic.reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0.)
            rot_loss = torch.tensor(0.)
            
        if self.learn_reid:
            reid_loss = self.loss_reid(target, output)
            pose_loss = self.loss_pose(target, output)
        else:
            reid_loss = torch.tensor(0.)
            pose_loss = torch.tensor(0.)

        center_factor = 1 / torch.exp(self.model.center_weight)
        center_loss_weight = center_factor * center_loss
        center_uncertainty_loss = self.model.center_weight

        offset_factor = 1 / torch.exp(self.model.offset_weight)
        offset_loss_weight = offset_factor * offset_loss
        offset_uncertainty_loss = self.model.offset_weight

        size_factor = 1 / torch.exp(self.model.size_weight)
        size_loss_weight = size_factor * size_loss
        size_uncertainty_loss = self.model.size_weight

        rot_factor = 1 / torch.exp(self.model.rot_weight)
        rot_loss_weight = rot_factor * rot_loss
        rot_uncertainty_loss = self.model.rot_weight

        tracking_factor = 1 / torch.exp(self.model.tracking_weight)
        tracking_loss_weight = tracking_factor * tracking_loss
        tracking_uncertainty_loss = self.model.tracking_weight
        
        reid_factor = 1 / torch.exp(self.model.reid_weight)
        reid_loss_weight = reid_factor * reid_loss
        reid_uncertainty_loss = self.model.reid_weight
        
        pose_factor = 1 / torch.exp(self.model.pose_weight)
        pose_loss_weight = pose_factor * pose_loss
        pose_uncertainty_loss = self.model.pose_weight

        # img loss
        center_img_loss = self.center_loss_fn(basic.sigmoid(center_img_e), center_img_g) / S

        loss_dict = {
            'center_loss': 10 * center_loss,
            'offset_loss': 10 * offset_loss,
            'tracking_loss': tracking_loss,
            'size_loss': size_loss,
            'rot_loss': rot_loss,
            'center_img': center_img_loss,
            
            'reid_loss': reid_loss,
            'pose_loss': pose_loss,
        }
        loss_weight_dict = {
            'center_loss': 10 * center_loss_weight,
            'offset_loss': 10 * offset_loss_weight,
            'tracking_loss': tracking_loss_weight,
            'size_loss': size_loss_weight,
            'rot_loss': rot_loss_weight,
            'center_img': center_img_loss,
            
            'reid_loss': reid_loss_weight,
            'pose_loss': pose_loss_weight,
        }
        stats_dict = {
            'center_uncertainty_loss': center_uncertainty_loss,
            'offset_uncertainty_loss': offset_uncertainty_loss,
            'tracking_uncertainty_loss': tracking_uncertainty_loss,
            'size_uncertainty_loss': size_uncertainty_loss,
            'rot_uncertainty_loss': rot_uncertainty_loss,
            
            'reid_uncertainty_loss': reid_uncertainty_loss,
            'pose_uncertainty_loss': pose_uncertainty_loss,
        }
        
        total_loss = sum(loss_weight_dict.values()) + sum(stats_dict.values())

        return total_loss, loss_dict
    
    def loss_reid(self, target, output):
        
        # info-nce loss
        bev_feat = output['instance_id_feat']        
        # get bev features with pid
        id_feat, pid, pid_counts = self.get_feature_vec_with_pid(bev_feat, target['pid_bev'])
        id_feat = F.normalize(id_feat, dim=1)
        
        if self.cont_type == 'simclr':
            sim_matrix = torch.matmul(id_feat, id_feat.t())
            logits = sim_matrix / self.temperature
            
            mask = torch.zeros_like(logits)
            for i, p in enumerate(pid):
                for j, p_self in enumerate(pid):
                    if p == p_self:
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0
            
            loss = (-logits * mask + torch.logsumexp(logits, dim=1, keepdim=True) * (1-mask)).mean()
            
        elif self.cont_type == 'moco':
            frames = target['frame']
            
            loss = torch.tensor(0., device=self.device)
            
            for i, (frame, num_pid) in enumerate(zip(frames, pid_counts)):
                prev_identities = self.load_identities(frame)
                if prev_identities is None:
                    continue
                else:
                    prev_identities.to(self.device)
                    identities = id_feat[pid_counts[:i].sum():pid_counts[:i+1].sum()]
                    assert identities.shape[0] == num_pid, 'num_pid should be the same'
                    assert identities.shape[1] == id_feat.shape[1], 'feature dim should be the same'
                    current_loss = F.mse_loss(identities, prev_identities).mean()
                    loss += current_loss
        
        return loss
    
    
    def loss_pose(self, target, output):
        
        pose_e = output['instance_pose'] # direction
        offset_g = target['offset_bev'] # velocity
        pose_g = offset_g[:, 2:]
        
        # normalize velocity
        ori_g = pose_g / (torch.norm(pose_g, dim=1, keepdim=True) + 1e-6)
        
        if self.learn_cont_pose: # contrastive loss
            pose_feat, _, _ = self.get_feature_vec_with_pid(pose_e, target['pid_bev'])
            
            pose_feat = F.normalize(pose_feat, dim=1)
            
            sim_matrix = torch.matmul(pose_feat, pose_feat.t())
            logits = sim_matrix / self.temperature_pose
            
            mask = (ori_g @ ori_g.t() + 1) / 2
            mask = mask * (mask > self.pose_cont_thresh).float()
            
            loss = (-logits * mask + torch.logsumexp(logits, dim=1, keepdim=True) * (1-mask)).mean()
        
        else: # pose guidance
            assert pose_e.shape[1] == 2, 'pose_e shape should be 2'
            valid_g = target['valid_bev']
            loss = torch.nn.functional.smooth_l1_loss(
                pose_e, ori_g, reduction='none').sum(dim=1, keepdim=True)
            loss = basic.reduce_masked_mean(loss, valid_g)
        
        return loss


    def check_distance(self, item, target):
        output, output_random  = \
            self.compute_bev(item)

        # get bev features with pid        
        feat, pid, _ = self.get_feature_vec_with_pid(output['instance_id_feat'], target['pid_bev'])
        feat_random, pid_random, _ = self.get_feature_vec_with_pid(output_random['instance_id_feat'], target['pid_bev_rand'])
        
        feat = F.normalize(feat, dim=1)
        feat_random = F.normalize(feat_random, dim=1)
        
        dist_self = 1 - F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=2)
        dist_random = 1 - F.cosine_similarity(feat.unsqueeze(1), feat_random.unsqueeze(0), dim=2)
        
        return dist_self, dist_random, pid, pid_random
            
    
    def get_feature_vec_with_pid(self, bev, pid):
        '''
        bev: B, C, H, W
        pid: B, 1, H, W
        return: bev_list, pid_list
        
        N persons in the bev
        pids contain the person id for each pixel
        -1: background
        0~K: person id
        '''
        
        # count pid numbers for each bev
        B = bev.shape[0]
        counts = []
        for i in range(B):
            unique_pid = torch.unique(pid[i].flatten())
            counts.append(len(unique_pid)-1)
        
        # filter out the background
        
        bev = bev.permute(0, 2, 3, 1)
        pid = pid.squeeze(1)
        bev = bev[pid != -1]
        pid = pid[pid != -1]
        
        bev = bev.reshape(-1, bev.shape[-1])
        pid = pid.reshape(-1)
        # print(bev.shape, pid.shape)
        # shape
        # bev: N, C
        # pid: N
        
        return bev, pid, counts


    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item, is_train=True)
        
        if batch_idx < 3:
            self.plot_data_train(item, target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)
        
        if self.learn_reid and self.cont_type == 'moco':
            self.store_identities(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss


    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item, is_train=True)

        # if batch_idx % 100 == 1:
        #     self.plot_data(target, output, batch_idx)
        self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)
        
        if self.learn_reid and self.cont_type == 'moco':
            self.store_identities(target, output)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss


    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)
        
        if self.test_mode == 'tracking':

            # output on bev plane
            center_e = output['instance_center']
            offset_e = output['instance_offset']
            size_e = output['instance_size']
            rot_e = output['instance_rot']
            
            
            self.draw_detection(item, output, batch_idx)

            xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
                center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
            )

            mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
            ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

            mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
            ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

            # detection
            for frame, grid_gt, xy, score in zip(item['frame'], item['grid_gt'], ref_xy, scores_e):
                frame = int(frame.item())
                valid = score > self.conf_threshold
                
                gt_list = [[frame, x.item(), y.item()] for x, y, _ in grid_gt[grid_gt.sum(1) != 0]]
                gt_list = np.array(gt_list)
                gt_list = gt_list[gt_list[:, 0] == frame]

                if len(gt_list) > 0:
                    self.moda_gt_list.extend(gt_list.tolist())
                self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])
                
            mota_now = []
            
            # tracking
            for seq_num, frame, grid_gt, bev_det, bev_prev, score, in (
                    zip(item['sequence_num'], item['frame'], item['grid_gt'], ref_xy.cpu(), ref_xy_prev.cpu(),
                        scores_e.cpu())):
                frame = int(frame.item())
                output_stracks = self.test_tracker.update(bev_det, bev_prev, score)
                
                mota_gt = [[seq_num.item(), frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                        for x, y, i in grid_gt[grid_gt.sum(1) != 0]]
                mota_pred = [[seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                                + s.xy.tolist() + [-1] for s in output_stracks]
                
                mota_gt = np.array(mota_gt)
                mota_pred = np.array(mota_pred)
                if len(mota_gt) == 0 or len(mota_pred) == 0:
                    mota_pred = np.zeros((0, 11))
                
                mota_gt = mota_gt[mota_gt[:, 0] == seq_num.item()]
                mota_pred = mota_pred[mota_pred[:, 0] == seq_num.item()]
                
                self.mota_gt_list.extend(mota_gt.tolist())
                self.mota_pred_list.extend(mota_pred.tolist())
        
                mota_now.extend(mota_pred.tolist())
            
            self.draw_prediction(item, output, mota_now, batch_idx)
            
                # self.mota_gt_list.extend([[seq_num.item(), frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                #                           for x, y, i in grid_gt[grid_gt.sum(1) != 0]])
                # self.mota_pred_list.extend([[seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                #                             + s.xy.tolist() + [-1]
                #                             for s in output_stracks])
        
        elif self.test_mode == 'feature_distance':
            item, target = batch
            
            dist_self, dist_random, pid, pid_random = \
                self.check_distance(item, target)
                
            for i, p in enumerate(pid):
                
                for j, p_self in enumerate(pid):
                    
                    if self.feat_dist_count_self[p, p_self] == 0:
                        self.feat_dist_self[p, p_self] = dist_self[i, j]
                    else:
                        self.feat_dist_self[p, p_self] = \
                            (self.feat_dist_self[p, p_self] * \
                            self.feat_dist_count_self[p, p_self] + dist_self[i, j]) / \
                            (self.feat_dist_count_self[p, p_self] + 1)
                    
                    self.feat_dist_count_self[p, p_self] += 1
                    
                # print(dist_prev.shape, pid_prev.shape)
                
                min_p_prev = pid[torch.argmin(dist_self[i])]
                
                if p == min_p_prev:
                    self.feat_correct_self += 1
                self.feat_total_self += 1
                    
                for j, p_random in enumerate(pid_random):
                        
                    if self.feat_dist_count_random[p, p_random] == 0:
                        self.feat_dist_random[p, p_random] = dist_random[i, j]
                    else:
                        self.feat_dist_random[p, p_random] = \
                            (self.feat_dist_random[p, p_random] * \
                            self.feat_dist_count_random[p, p_random] + dist_random[i, j]) / \
                            (self.feat_dist_count_random[p, p_random] + 1)
                    
                    self.feat_dist_count_random[p, p_random] += 1
                    
                min_p_random = pid_random[torch.argmin(dist_random[i])]
                if p == min_p_random:
                    self.feat_correct_random += 1
                self.feat_total_random += 1
        
        elif self.test_mode == 'save_features':
            item, target = batch
            
            feat = output['instance_id_feat']
            pid = target['pid_bev']
            
            feat, pid, counts = self.get_feature_vec_with_pid(feat, pid)
            
            for i, count in enumerate(counts):
                self.feat_dist_list.append(feat[pid_counts[:i].sum():pid_counts[:i+1].sum()])


    def on_test_epoch_end(self):
        
        if self.test_mode == 'tracking':
        
            log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'
            
            # detection
            '''
            unit: 2.5cm for wildtrack, multiviewx
                    1cm for aicity_lt, aicity
            '''
            unit = 2.5 if self.test_dataset == 'wildtrack' or self.test_dataset == 'multiviewx' \
                else 1. if self.test_dataset == 'aicity_lt' or self.test_dataset == 'aicity' \
                else 1.
            # detection
            pred_path = osp.join(log_dir, 'moda_pred.txt')
            gt_path = osp.join(log_dir, 'moda_gt.txt')
            np.savetxt(pred_path, np.array(self.moda_pred_list), '%f', delimiter=' ', newline='\n')
            np.savetxt(gt_path, np.array(self.moda_gt_list), '%d', delimiter=' ', newline='\n')
            recall, precision, moda, modp = modMetricsCalculator(osp.abspath(pred_path), osp.abspath(gt_path), unit)
            self.log(f'detect/recall', recall)
            self.log(f'detect/precision', precision)
            self.log(f'detect/moda', moda)
            self.log(f'detect/modp', modp)

            # tracking
            '''
            unit: 2.5cm for wildtrack, multiviewx
                    1cm for aicity_lt, aicity
            '''
            scale = 0.025 if self.test_dataset == 'wildtrack' or self.test_dataset == 'multiviewx' \
                else 0.01 if self.test_dataset == 'aicity_lt' or self.test_dataset == 'aicity' \
                else 1.
            pred_path = osp.join(log_dir, 'mota_pred.txt')
            gt_path = osp.join(log_dir, 'mota_gt.txt')
            np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')
            np.savetxt(gt_path, np.array(self.mota_gt_list), '%f', delimiter=',')
            summary = mot_metrics(osp.abspath(pred_path), osp.abspath(gt_path), scale)
            summary = summary.loc['OVERALL']
            for key, value in summary.to_dict().items():
                if value >= 1 and key[:3] != 'num':
                    value /= summary.to_dict()['num_unique_objects']
                value = value * 100 if value < 1 else value
                value = 100 - value if key == 'motp' else value
                self.log(f'track/{key}', value)
        
        elif self.test_mode == 'feature_distance':
            log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'
            
            acc_self = self.feat_correct_self / self.feat_total_self
            acc_random = self.feat_correct_random / self.feat_total_random
            
            self.log('acc_self', acc_self)
            self.log('acc_random', acc_random)
            
            # print(f'prev: {acc_prev}, random: {acc_random}, feat: {acc_feat}')
            # clip pids
            count_sum = self.feat_dist_count_self.sum(1) + self.feat_dist_count_random.sum(1)
            
            max_pid = torch.argmin(count_sum)
            # print(count_sum, max_pid)
            
            self.feat_dist_self = self.feat_dist_self[:max_pid, :max_pid]
            self.feat_dist_random = self.feat_dist_random[:max_pid, :max_pid]
            
            # save plots to tensorboard in eval loop
            
            # set dist to 2 if not calculated
            self.feat_dist_self[self.feat_dist_self == -1] = 2
            self.feat_dist_random[self.feat_dist_random == -1] = 2
            
            # normalize to 0 to 1
            self.feat_dist_self /= 2.
            self.feat_dist_random /= 2.
            
            # to numpy
            dist_self = self.feat_dist_self.detach().cpu().numpy()
            dist_random = self.feat_dist_random.detach().cpu().numpy()
            
            writer = self.logger.experiment
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            ax1.imshow(dist_self)
            ax2.imshow(dist_random)
            
            ax1.set_title('self')
            ax2.set_title('random')
            
            plt.tight_layout()
            writer.add_figure(f'plot/feature_distance', fig, global_step=self.global_step)
            plt.close(fig)


    def predict_step(self, batch, batch_idx):
        item, _ = batch
        output = self(item)

        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

        # detection
        for frame, xy, score in zip(item['frame'], ref_xy, scores_e):
            frame = int(frame.item())
            valid = score > self.conf_threshold
            self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])
            
        mota_now = []
        
        # tracking
        for seq_num, frame, bev_det, bev_prev, score, in (
                zip(item['sequence_num'], item['frame'], ref_xy.cpu(), ref_xy_prev.cpu(),
                    scores_e.cpu())):
            frame = int(frame.item())
            output_stracks = self.test_tracker.update(bev_det, bev_prev, score)
            
            mota_pred = [[seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                            + s.xy.tolist() + [-1] for s in output_stracks]
            
            mota_pred = np.array(mota_pred)
            if len(mota_pred) == 0:
                mota_pred = np.zeros((0, 11))

            mota_pred = mota_pred[mota_pred[:, 0] == seq_num.item()]
            self.mota_pred_list.extend(mota_pred.tolist())
            
            mota_now.extend(mota_pred.tolist())
        
        self.draw_prediction(item, output, mota_now)
        
        return self.moda_pred_list, self.mota_pred_list


    def on_predict_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f', delimiter=' ', newline='\n')

        # tracking
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')


    def plot_data(self, target, output, batch_idx=0):
        
        writer = self.logger.experiment
        
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
        
        
    def draw_detection(self, item, output, batch_idx=0):
        
        writer = self.logger.experiment
        
        center_e: torch.Tensor = output['instance_center'][0]
        rgb_cams: torch.Tensor = item['img'][0]
        pix_T_cams: torch.Tensor = item['intrinsic'][0]
        cams_T_global: torch.Tensor = item['extrinsic'][0]
        ref_T_global: torch.Tensor = item['ref_T_global'][0]
        
        S = rgb_cams.shape[0]
        heatmap = center_e.amax(0).sigmoid().squeeze().cpu().unsqueeze(0).unsqueeze(0).repeat(S, 1, 1, 1)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(900, 900), mode='bilinear', align_corners=False)
     
        ref_T_cams = torch.matmul(ref_T_global.detach().cpu().repeat(S, 1, 1), 
                                  torch.inverse(cams_T_global.detach().cpu()))  # B*S,4,4
        cams_T_ref = torch.inverse(ref_T_cams) # B*S,4,4
        pix_T_ref = torch.matmul(pix_T_cams.detach().cpu()[:, :3, :3], cams_T_ref[:, :3, [0, 1, 3]])  # B*S,3,3
        # ref_T_pix = torch.inverse(pix_T_ref) # B*S,3,3
        
        # warp heatmap to image
        rgb_cams = torch.nn.functional.interpolate(rgb_cams.detach().cpu(), size=(720, 1280), mode='bilinear', align_corners=False)
        rgb_cams = rgb_cams.permute(0, 2, 3, 1).numpy()
        warped_heatmap = kornia.geometry.warp_perspective(heatmap, pix_T_ref, (720, 1280)).permute(0, 2, 3, 1).squeeze(-1).numpy()
        
        heatmap_colored = plt.get_cmap('jet')(warped_heatmap)[:, :, :, :3]  # Drop the alpha channel
        mixed = 0.4 * rgb_cams + 0.6 * heatmap_colored

        fig, axes = plt.subplots(1, S, figsize=(12, 8))
        for cam in range(S):
            ax = axes[cam]
            ax.imshow(mixed[cam])
            ax.set_title(f'cam_{cam+1}')
        plt.tight_layout()
        writer.add_figure(f'det/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)

    
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


    def draw_prediction(self, item, output, mota_now, batch_idx=0):
        
        writer = self.logger.experiment
        
        center_e: torch.Tensor = output['instance_center'][0]
        rgb_cams: torch.Tensor = item['img'][0]
        pix_T_cams: torch.Tensor = item['intrinsic'][0]
        cams_T_global: torch.Tensor = item['extrinsic'][0]
        ref_T_global: torch.Tensor = item['ref_T_global'][0]
        
        # print(center_e.shape, rgb_cams.shape, pix_T_cams.shape, cams_T_global.shape, ref_T_global.shape)
        
        S = rgb_cams.shape[0]
     
        ref_T_cams = torch.matmul(ref_T_global.detach().cpu().repeat(S, 1, 1), 
                                  torch.inverse(cams_T_global.detach().cpu()))  # S,4,4
        cams_T_ref = torch.inverse(ref_T_cams) # B*S,4,4
        pix_T_ref = torch.matmul(pix_T_cams.detach().cpu()[:, :3, :3], cams_T_ref[:, :3, [0, 1, 3]])  # S,3,3
        
        mota_data = np.asarray(mota_now)
        
        mota_data = mota_data[:, (1, 2, 8, 9)]
        mota_world = torch.from_numpy(np.concatenate([mota_data[:, 2:], 
                                       np.ones((mota_data.shape[0], 1))], axis=1)).float() # B,3
        mota_world = mota_world.unsqueeze(0).repeat(S, 1, 1).unsqueeze(-1) # S,B,3,1
        
        pix_T_ref = pix_T_ref.unsqueeze(1).repeat(1, mota_world.shape[1], 1, 1) # S,B,3,3
        
        mota_pix = torch.matmul(pix_T_ref, mota_world).squeeze(-1) # S,B,3
        mota_pix = mota_pix[:, :, :2] / mota_pix[:, :, 2].unsqueeze(-1) # S,B,2
        
        mota_data = torch.from_numpy(mota_data).float()
        mota_data = mota_data.unsqueeze(0).repeat(S, 1, 1) # S,B,4
        mota_data = torch.cat([mota_data, mota_pix], dim=-1) # S,B,6
        
        ids = mota_data[:, :, 2].unique()
        
        colors = {int(id): mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[int(idx)]]
                  for idx, id in enumerate(ids.flatten().tolist())}
        
        cams = {}
        
        for cam_idx in range(S):
            img = rgb_cams[cam_idx].detach().permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (1280, 720))
            
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            
            for id in ids:
                data = mota_data[cam_idx, mota_data[cam_idx, :, 2] == id]
                for i in range(data.shape[0]):
                    x, y = data[i, 4], data[i, 5]
                    re_x, re_y = data[i, 2], data[i, 3]
                    re_x, re_y = (re_x-450)/100., (re_y-450)/100.
                    
                    if x < 0 or x > 1280 or y < 0 or y > 720:
                        continue
                    
                    draw.text((x, y), f'person{int(id)}', fill=colors[int(id)])
                    draw.text((x, y+10), f'(x={re_x:.2f}m, y={re_y:.2f}m)', fill=colors[int(id)])
                    draw.ellipse((x-5, y-5, x+5, y+5), fill=colors[int(id)])
            
            img = np.array(img)
            cams[cam_idx] = img
        
        # draw mosaic
        # mosaic tiles
        r = np.ceil(np.sqrt(S)).astype(int)
        mosaic = np.zeros((r*720, r*1280, 3), dtype=np.uint8)
        for cam_idx in range(S):
            x = cam_idx % r
            y = cam_idx // r
            mosaic[y*720:(y+1)*720, x*1280:(x+1)*1280] = cams[cam_idx]
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(mosaic)
        plt.axis('off')
        plt.tight_layout()
        writer.add_figure(f'predict/{batch_idx}', fig, global_step=self.global_step)
        
        
if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")
            parser.link_arguments("data.init_args.data_dir", "model.test_dataset_dir")


    cli = MyLightningCLI(WorldTrackModel)
