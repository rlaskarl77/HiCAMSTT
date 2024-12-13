from datetime import datetime
import os
import os.path as osp
import time
import argparse
import yaml
import random
import json
from typing import Dict, List   
import torch
import lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw
import numpy as np
from kornia.geometry import warp_perspective
import cv2

from torch import nn

from models import Segnet, MVDet, Liftnet, Bevformernet, MVDetr
from models.loss import FocalLoss, compute_rot_loss
from tracking.multitracker import JDETracker
from utils import vox, basic, decode
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics
from utils.annotation import ObjectType, Data, Camera

from datasets import StreamFactoryDataModule


def normalize_color(color):
    """
    Normalize a (B, G, R) or (R, G, B) OpenCV color to a matplotlib RGB color.
    """
    if isinstance(color, tuple) and len(color) == 3:
        # Normalize color values to [0, 1] and convert to matplotlib RGB format
        normalized_color =  tuple(c / 255.0 for c in color)
        return tuple(normalized_color)
    else:
        raise ValueError(f"Invalid color format: {color}")


def reproject_world_to_image(world_coords, intrinsic, extrinsic):
    if isinstance(world_coords, torch.Tensor):
        world_coords = world_coords.cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.cpu().numpy()

    world_coords = world_coords.astype(np.float32)
    intrinsic = intrinsic.astype(np.float32)
    extrinsic = extrinsic.astype(np.float32)

    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, np.array([0, 0, 0, 1])])
    if world_coords.ndim == 1:
        world_coords = world_coords.reshape(1, -1)
    if world_coords.shape[1] == 2:
        world_coords = np.hstack((world_coords, np.zeros((world_coords.shape[0], 1)), np.ones((world_coords.shape[0], 1))))
    elif world_coords.shape[1] == 3:
        world_coords = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    
    camera_coords_hom = (extrinsic @ world_coords.T).T
    image_coords_hom = (intrinsic[:3, :3] @ camera_coords_hom[:, :3].T).T
    image_coords = image_coords_hom[:, :2] / image_coords_hom[:, 2:]
    return image_coords


class WorldTrackModel(pl.LightningModule):
    def __init__(
            self,
            model_name='mvdet',
            encoder_name='res18',
            learning_rate=0.001,
            resolution=(250, 2, 125),
            bounds=(0, 500, 0, 1000, 0, 2),
            num_cameras=2,
            depth=(32, 250, 3250),
            scene_centroid=(0.0, 0.0, 0.0),
            num_classes=1,
            z_sign=1,
            feat2d_dim=128,
            # decoder
            learn_reid=False,
            learn_pose=False,
            id_pose_decompose=False,
            reid_feat=128,
            pose_feat=128,
            hard_mask=False,
            temperature=1.,
            cont_type='simclr',
            learn_cont_pose=True,
            temperature_pose=1.,
            pose_cont_thresh=0.5,
            # tracker
            use_reid_tracking=False,
            use_temporal_cache=True,
            max_detections=60,
            conf_threshold=0.5,
            max_cache=32,
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
            # DDP setting for CL
            ddp_contrastive=False,
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
        self.learn_pose = learn_pose
        self.decoder_args = {
            "learn_reid": learn_reid,
            "learn_pose": learn_pose,
            "id_pose_decompose": id_pose_decompose,
            "reid_feat": reid_feat,
            "pose_feat": pose_feat,
            "hard_mask": hard_mask,
        }
        # contrastive loss
        self.temperature = temperature
        self.cont_type = cont_type # simclr or moco
        self.learn_cont_pose = learn_cont_pose
        self.pose_cont_thresh = pose_cont_thresh
        self.temperature_pose = temperature_pose
        
        if self.cont_type == 'moco':
            self.moco_memory_bank = dict()
        
        assert pose_feat == 4 or (learn_pose and learn_cont_pose) or (not learn_pose), \
            'pose_feat should be 4 (rot, val) if learn_pose is False'
        
        # Tracker
        self.use_reid_tracking = use_reid_tracking
        self.tracker_args = {
            "use_reid_tracking": use_reid_tracking,
            "conf_thres": conf_threshold,
            "track_buffer": track_buffer,
            "lapjv_thresh": lapjv_thresh,
            "lapjv_thresh2": lapjv_thresh2,
            "max_spatial_dist": max_spatial_dist,
            "max_spatial_dist2": max_spatial_dist2,
            "dist_alpha": dist_alpha,
            "temp_mixing": temp_mixing,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        }
        self.test_tracker = JDETracker(**self.tracker_args)

        # Loss
        self.center_loss_fn = FocalLoss()
        
        self.geometric_loss_fn = torch.nn.functional.smooth_l1_loss

        # Temporal cache
        self.frame_counter = 0
        self.use_temporal_cache = use_temporal_cache
        if self.use_temporal_cache:
            self.prev_frame_feat = None

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
        
        
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


    def forward(self, item):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        ref_T_global: (B,4,4)
        vox_util: vox util object
        """
        prev_bev = self.load_cache(item['time'])

        output = self.model(
            rgb_cams=item['img'].to(self.device),
            pix_T_cams=item['intrinsic'].to(self.device),
            cams_T_global=item['extrinsic'].to(self.device),
            ref_T_global=item['ref_T_global'].to(self.device),
            vox_util=self.vox_util,
            prev_bev=prev_bev,
        )

        if self.use_temporal_cache:
            self.store_cache(item['time'], output['bev_raw'].clone().detach())

        return output

    def load_cache(self, _):  # Ignore timestamp parameter
        """Load previous frame's features if available"""
        if not self.use_temporal_cache or self.prev_frame_feat is None:
            return None
        return self.prev_frame_feat.unsqueeze(0)

    def store_cache(self, _, bev_feat):  # Ignore timestamp parameter
        """Store current frame's features"""
        if not self.use_temporal_cache:
            return
        try:
            self.prev_frame_feat = bev_feat[0].clone()  # Only store first batch item
        except:
            pass

    def predict_step(self, batch, batch_idx):
        self.frame_counter += 1  # Increment frame counter
        
        item = batch
        output = self(item)
        
        self.moda_now, self.mota_now = [], []

        # ref_T_global = item['ref_T_global']
        # global_T_ref = torch.inverse(ref_T_global)
        
        # try:
        #     self.draw_detection(item, output, batch_idx)
        # except Exception as e:
        #     print(e)

        # output on bev plane
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        # size_e = output['instance_size']
        # rot_e = output['instance_rot']
        size_e = None
        rot_e = None
        

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

        # detection
        for timestamp, xy, score in zip(item['time'], ref_xy, scores_e):
            valid = score > self.conf_threshold
            self.moda_now.extend([[self.frame_counter, x.item(), y.item()] for x, y in xy[valid]])

        # tracking
        for seq_num, timestamp, bev_det, bev_prev, score, in (
                zip(item['sequence_num'], item['time'], ref_xy.cpu(), ref_xy_prev.cpu(),
                    scores_e.cpu())):
            output_stracks = self.test_tracker.update(bev_det, bev_prev, score)
            
            # Convert tracked points from reference to global coordinates
            global_T_ref = torch.inverse(item['ref_T_global'][0])  # 4,4
            
            updated_stracks = []
            for track in output_stracks:
                # Convert xy to homogeneous coordinates
                xy_ref = torch.tensor([track.xy[0], track.xy[1], 0, 1], dtype=torch.float32)
                # Transform to global
                xy_global = torch.matmul(global_T_ref, xy_ref)
                track_xy = xy_global[:2].tolist()
                updated_stracks.append(track_xy)
            
            self.mota_now.extend([[seq_num, self.frame_counter, s.track_id, -1, -1, -1, -1, s.score.item()]
                                + p + [-1]
                                for s, p in zip(output_stracks, updated_stracks)])
        
        if len(self.mota_now) == 0:
            self.mota_now = np.zeros((0, 10))
        if len(self.moda_now) == 0:
            self.moda_now = np.zeros((0, 3))
        
        
            
            
    def log_results(self, time):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'

        # detection
        pred_path = osp.join(log_dir, f'{time}_moda.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f')

        pred_path = osp.join(log_dir, f'{time}_mota.txt')
        np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')
        
        hdc_data = self.convert_mota_to_hdc_format(self.mota_pred_list, time)
        hdc_data.save_to_file(f"SNU_{datetime.fromisoformat(time).strftime('%Y_%m_%d-%H-%M-%S-%f')[:-3]}_1.json")

         
    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        np.savetxt(pred_path, np.array(self.moda_now), '%f')

        pred_path = osp.join(log_dir, 'mota_pred.txt')
        np.savetxt(pred_path, np.array(self.mota_now), '%f', delimiter=',')

    def convert_mota_to_hdc_format(self, mota_pred_list, time):
        data = np.asarray(mota_pred_list)
        if len(data) == 0:
            return Data(time=time, camera=[])
        
        data = data[:, (1, 2, 8, 9)]
        object_list = []
        for frame, track_id, x, y in data:
            object_list.append(ObjectType(
                    type=0,
                    id=track_id,
                    action=0,
                    value=0,
                    posx=x/100,
                    posy=0.,
                    posz=y/100,
                    sizex=0,
                    sizey=0,
                    sizez=0,
                    execution=0
                ))  

        hdc_data = Data(
                time=time,
                camera=[
                    Camera(
                        camera_id="-1",
                        object_type=object_list)])
        return hdc_data
    
    def draw_detection(self, item, output, batch_idx=0):
        
        writer = self.logger.experiment
        
        center_e: torch.Tensor = output['instance_center'][0]
        rgb_cams: torch.Tensor = item['img'][0]
        pix_T_cams: torch.Tensor = item['intrinsic'][0]
        cams_T_global: torch.Tensor = item['extrinsic'][0]
        ref_T_global: torch.Tensor = item['ref_T_global'][0]
        
        # print(center_e.shape, rgb_cams.shape, pix_T_cams.shape, cams_T_global.shape, ref_T_global.shape)
        
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
        warped_heatmap = warp_perspective(heatmap, pix_T_ref, (720, 1280)).permute(0, 2, 3, 1).squeeze(-1).numpy()
        
        heatmap_colored = plt.get_cmap('jet')(warped_heatmap)[:, :, :, :3]  # Drop the alpha channel
        mixed = 0.4 * rgb_cams + 0.6 * heatmap_colored
        
        input_img = (np.concatenate([rgb_cams[0], rgb_cams[1], rgb_cams[2]], axis=1) * 255).astype(np.uint8)
        input_img = cv2.resize(input_img, (1920, 360))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        mosaic = np.concatenate([mixed[0], mixed[1], mixed[2]], axis=1)
        mosaic = (mosaic * 255).astype(np.uint8)
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        mosaic_resized = cv2.resize(mosaic, (1920, 360))
        
        cv2.imshow('Input', input_img)
        cv2.imshow('Mosaic', mosaic_resized)
        cv2.waitKey(1)
    
    def draw_prediction(self, item, output, mota_now):
        
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
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cams[cam_idx] = img
        
        # draw mosaic
        mosaic = np.concatenate([cams[0], cams[1], cams[2]], axis=1)
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Mosaic', mosaic)
        cv2.waitKey(1)
        

class WorldTrackInference:
    def __init__(self, model_configs, sources, save_dir='.', 
                 max_id=999999, visualize=False):
        self.models = {}
        self.datamodule = StreamFactoryDataModule(sources=sources, verbose=False)
        self.global_id_counter = 0
        self.max_id = max_id
        self.id_mapping = {}
        self.id_history = {}
        self.save_dir = save_dir
        self.visualize = visualize
        
        for scene_name, config in model_configs.items():
            checkpoint_path = config.pop('checkpoint_path')
            model = WorldTrackModel(**config)
            model.load_from_checkpoint(checkpoint_path)
            # model = WorldTrackModel.load_from_checkpoint(
            #     checkpoint_path=checkpoint_path,
            #     **config
            # )
            model.eval()
            model.cuda()
            self.models[scene_name] = model
        self.color_map = {}
        
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def get_next_available_id(self):
        """Find the next available ID, reusing old ones if needed"""
        if self.global_id_counter <= self.max_id:
            next_id = self.global_id_counter
            self.global_id_counter += 1
            return next_id
            
        # Find the smallest unused ID
        used_ids = set(self.id_mapping.values())
        for i in range(self.max_id + 1):
            if i not in used_ids:
                return i
                
        # If no IDs available, reuse oldest ID
        oldest_time = float('inf')
        oldest_id = 0
        for id, data in self.id_history.items():
            if data['last_seen'] < oldest_time:
                oldest_time = data['last_seen']
                oldest_id = id
        
        # Clean up old ID
        self.id_history.pop(oldest_id)
        self.id_mapping = {k: v for k, v in self.id_mapping.items() if v != oldest_id}
        return oldest_id

    def run_inference(self):
        self.datamodule.setup('predict')
        
        
        print("Waiting for all streams to initialize...")
        for scene_name, event in self.datamodule.init_events.items():
            event.wait()
        print("All streams initialized. Starting data loading.")
        
        loader = self.datamodule.predict_dataloader()
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        while True:
            self.starter.record()
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
            scene_results = []
            all_scene_data = []
            
            try:
                # Get synchronized data from all scenes
                scene_data = next(loader)
                
                if not scene_data:
                    print("No data received. Skipping inference.")
                    time.sleep(0.1)
                    continue
                
                # Process each scene's data
                for scene_name, batch in scene_data.items():
                    model = self.models[scene_name]
                    
                    try:
                        with torch.no_grad():
                            output = model.predict_step(batch, 0)
                            results = model.mota_now
                            
                            if len(results) > 0:
                                results = np.array(results)
                                scene_results.append((scene_name, results))
                                
                                all_scene_data.append({
                                    'images': batch['img'][0].cpu().numpy(),
                                    'results': np.array(results),
                                    'intrinsics': batch['intrinsic'][0].cpu().numpy(),
                                    'extrinsics': batch['extrinsic'][0].cpu().numpy()
                                })
                            else:
                                all_scene_data.append({
                                    'images': batch['img'][0].cpu().numpy(),
                                    'results': np.zeros((0, 10)),
                                    'intrinsics': batch['intrinsic'][0].cpu().numpy(),
                                    'extrinsics': batch['extrinsic'][0].cpu().numpy()
                                })
                                
                    except Exception as e:
                        print(f"Error processing scene {scene_name}: {e}")
                        continue
                
                if scene_results:
                    final_results = self.process_all_results(scene_results)
                    
                    if len(final_results) > 0:
                        hdc_data = self.convert_results_to_hdc_format(final_results, current_time)
                        save_path = os.path.join(self.save_dir, f"SNU_{current_time}_1.json")
                        hdc_data.save_to_file(save_path)

                    save_path = os.path.join(self.save_dir, "visualizations")
                    os.makedirs(save_path, exist_ok=True)
                        
                    if self.visualize:
                        self.visualize_all_results(all_scene_data)
                
                self.ender.record()
                torch.cuda.synchronize()
                elapsed_time = self.starter.elapsed_time(self.ender)
                print(f'Total inference time: {elapsed_time:.2f} ms')
                
            except StopIteration:
                continue
            except Exception as e:
                print(f"Error in inference loop: {e}")
                continue

    
    def prepare_image_for_opencv(self, image):
        # Convert PyTorch Tensor to NumPy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        # Ensure image is a NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a NumPy array or a PyTorch Tensor.")
        # Rescale to 0-255 and convert to uint8 if dtype is not uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)  # Clamp values to 0-1 for normalization
            image = (image * 255).astype(np.uint8)
        # Ensure H x W x C format
        if len(image.shape) == 3 and image.shape[0] == 3:  # C x H x W to H x W x C
            image = np.transpose(image, (1, 2, 0))
        elif len(image.shape) != 3 or image.shape[2] != 3:  # Ensure 3 channels
            raise ValueError(f"Image shape {image.shape} is invalid. Expected H x W x C with 3 channels.")
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        return image


    def generate_color_map(self, object_ids):
        """
        Generate a consistent color map for object IDs.

        Args:
            object_ids (list[int]): List of object IDs.

        Returns:
            dict: A dictionary mapping object IDs to RGB color tuples.
        """
        for obj_id in object_ids:
            if obj_id not in self.color_map:
                self.color_map[obj_id] = tuple(random.randint(0, 255) for _ in range(3))
        return self.color_map

    def visualize_all_results(self, all_scene_data, save_path):
        """
        Visualizes tracking results with world coordinates and image coordinates,
        and saves the visualizations.

        Args:
            all_scene_data (list): List of scene data dictionaries with images, results, intrinsics, and extrinsics.
            save_path (str): Directory to save the visualizations.
        """
        os.makedirs(save_path, exist_ok=True)

        # Object IDs and color mapping for visualization
        pred_object_ids = []
        for scene_data in all_scene_data:
            if len(scene_data['results']) > 0:
                pred_object_ids.extend(scene_data['results'][:, 2].astype(int))
        pred_object_ids = np.unique(pred_object_ids)
        pred_color_map = self.generate_color_map(pred_object_ids)

        # Initialize trajectories for visualization
        if not hasattr(self, "trajectories_pred"):
            self.trajectories_pred = {obj_id: [] for obj_id in pred_object_ids}

        for obj_id in pred_object_ids:
            if obj_id not in self.trajectories_pred:
                self.trajectories_pred[obj_id] = []

        # Prepare 5x3 grid for visualization (2 columns for cameras, 1 for world grid)
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(5, 3, width_ratios=[1, 1, 1])  # 5 rows, 3 columns

        # Process each scene
        for scene_idx, scene_data in enumerate(all_scene_data):
            images = scene_data["images"]
            intrinsic = scene_data["intrinsics"]
            extrinsic = scene_data["extrinsics"]
            results = scene_data["results"]

            for cam_idx in range(len(images)):
                ax = fig.add_subplot(gs[scene_idx, cam_idx])
                img = images[cam_idx].copy()
                cam_image = self.prepare_image_for_opencv(img)
                # Draw projected image coordinates
                for obj_id in pred_object_ids:
                    obj_data = results[results[:, 2] == obj_id]
                    if obj_data.shape[0] == 0:
                        continue

                    world_coords = obj_data[:, [8, 9]]

                    # Convert to homogeneous coordinates for re-projection
                    homogeneous_coords = np.hstack((world_coords[:, :2], np.zeros((world_coords.shape[0], 1)), np.ones((world_coords.shape[0], 1))))
                    image_coords = reproject_world_to_image(homogeneous_coords, intrinsic[cam_idx], extrinsic[cam_idx])

                    # Draw points on the image
                    for coord in image_coords:
                        x, y = int(coord[0]), int(coord[1])
                        cv2.circle(cam_image, (x, y), 10, pred_color_map[obj_id], -1)
                        text_position = (x, y - 10)  # Position text slightly above the circle
                        cv2.putText(
                            cam_image, f"ID {obj_id}", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color_map[obj_id], 2
                        )

                # Display the image with annotations
                ax.imshow(cam_image)
                ax.set_title(f"Scene {scene_idx} - Cam {cam_idx}")
                ax.axis("off")

        # Process world coordinates for all object IDs for the current frame
        ax_world = fig.add_subplot(gs[:, 2])  # Use the last column for the world grid
        ax_world.set_title("World Coordinates")
        ax_world.set_xlim(0, 25)  # Reverse x-axis
        ax_world.set_ylim(0, 290)  # Reverse y-axis
        ax_world.set_aspect("equal")
        ax_world.grid(True)

        for obj_id in pred_object_ids:
            # Filter only current frame and current object
            for scene_data in all_scene_data:
                results = scene_data["results"]
                obj_results = results[results[:, 2] == obj_id]
                if obj_results.shape[0] == 0:
                    continue

                # Extract world coordinates for current frame
                world_coords = obj_results[:, [8, 9]]

                # Plot current world coordinates
                ax_world.plot(world_coords[:, 0], world_coords[:, 1], 'o', label=f"ID {obj_id}", color=normalize_color(pred_color_map[obj_id]))
                ax_world.text(world_coords[-1, 0], world_coords[-1, 1] + 0.5, f"ID {obj_id}", fontsize=10, color=normalize_color(pred_color_map[obj_id]))

        ax_world.legend(loc="upper right", fontsize=9)

        # Save the combined visualization
        frame_save_path = os.path.join(save_path, f"frame_{time.time():.2f}.png")
        plt.tight_layout()
        plt.savefig(frame_save_path)
        plt.close(fig)




    def process_all_results(self, scene_results):
        all_detections = []
        
        # Collect all detections with scene info
        for scene_name, results in scene_results:
            for result in results:
                det = result.copy()
                local_id = int(det[2])
                scene_track_id = (scene_name, local_id)
                det = np.append(det, [scene_name, local_id])  # Add scene info
                all_detections.append(det)
        
        if not all_detections:
            return []
            
        all_detections = np.array(all_detections)
        return self.nms_tracking_results(all_detections)

    def nms_tracking_results(self, results, distance_threshold=2.0):
        """
        NMS with ID preservation
        results: Nx12 array (seq, frame, track_id, _, _, _, _, score, x, y, scene_name, local_id)
        """
        if len(results) == 0:
            return results

        # Get numeric data only for distance calculation
        numeric_data = results[:, :10].astype(float)  # Exclude scene_name, local_id
        scene_info = results[:, 10:]  # Keep scene info separately
        
        # Group by proximity
        groups = []
        used = set()
        
        for i in range(len(results)):
            if i in used:
                continue
                
            current_group = [i]
            current_pos = numeric_data[i, 8:10]  # Use numeric data for position
            
            for j in range(i + 1, len(results)):
                if j in used:
                    continue
                    
                other_pos = numeric_data[j, 8:10]  # Use numeric data for position
                distance = np.sqrt(np.sum((other_pos - current_pos) ** 2))
                
                if distance < distance_threshold:
                    current_group.append(j)
                    used.add(j)
            
            groups.append(current_group)
            used.add(i)

        final_results = []
        for group in groups:
            group_results = results[group]
            
            # Get scene track IDs in the group
            scene_track_ids = [(r[-2], int(r[-1])) for r in group_results]
            
            # Find existing global ID
            global_id = None
            for scene_track_id in scene_track_ids:
                if scene_track_id in self.id_mapping:
                    global_id = self.id_mapping[scene_track_id]
                    break
            
            # Create new global ID if needed
            if global_id is None:
                global_id = self.get_next_available_id()
            
            # Update mappings for all tracks in group
            for scene_track_id in scene_track_ids:
                self.id_mapping[scene_track_id] = global_id
            
            # Use highest scoring detection's position
            best_idx = np.argmax(group_results[:, 7])
            best_detection = group_results[best_idx].copy()
            best_detection[2] = global_id  # Update with global ID
            
            # Update position history
            pos = best_detection[8:10]
            self.id_history[global_id] = {
                'pos': pos,
                'last_seen': time.time()
            }
            
            final_results.append(best_detection[:10])  # Remove scene info

        # Clean old entries
        current_time = time.time()
        old_ids = {id for id, data in self.id_history.items() 
                  if current_time - data['last_seen'] > 5.0}
        
        for id in old_ids:
            self.id_history.pop(id)
            self.id_mapping = {k: v for k, v in self.id_mapping.items() if v != id}

        return np.array(final_results)

    @staticmethod
    def convert_results_to_hdc_format(results, time):
        data = np.asarray(results)
        if len(data) == 0:
            return Data(time=time, camera=[])
        
        data = data[:, (1, 2, 8, 9)]
        object_list = []
        for frame, track_id, x, y in data:
            object_list.append(ObjectType(
                type=0,
                id=int(track_id),
                action=0,
                value=0,
                posx=float(x),
                posy=0.,
                posz=float(y),
                sizex=0,
                sizey=0,
                sizez=0,
                execution=0
            ))  

        return Data(
            time=time,
            camera=[Camera(camera_id=100, objects=object_list)]
        )

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict) -> None:
    """Validate config structure and required fields"""
    required_model_fields = ['checkpoint_path', 'resolution', 'bounds', 'scene_centroid', 'num_cameras']
    
    if 'model_configs' not in config or 'sources' not in config:
        raise ValueError("Config must contain 'model_configs' and 'sources'")
        
    if len(config['sources']) != 10:  # Assuming we need exactly 10 sources
        raise ValueError("Config must contain exactly 10 sources")
        
    for scene, cfg in config['model_configs'].items():
        for field in required_model_fields:
            if field not in cfg:
                raise ValueError(f"Missing required field '{field}' in {scene} config")

def parse_args():
    parser = argparse.ArgumentParser(description='Run multi-scene tracking inference')
    parser.add_argument('--config', type=str, required=True,
                      default='example_factory_test_config.yml',
                      help='Path to config YAML file')
    parser.add_argument('--save-dir', type=str, default='.',
                      help='Directory to save output JSON files')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable visualization')
    return parser.parse_args()

if __name__ == '__main__':
    import signal
    import sys
    
    args = parse_args()
    
    # Load and validate config
    config = load_config(args.config)
    validate_config(config)
    
    torch.set_float32_matmul_precision('medium')
    
    inference = WorldTrackInference(
        model_configs=config['model_configs'],
        sources=config['sources'],
        save_dir=args.save_dir,
        max_id=999999,
        visualize=args.visualize,
    )
    
    def signal_handler(sig, frame):
        print('Shutting down...')
        inference.datamodule.teardown('predict')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        inference.run_inference()
    except Exception as e:
        print(f"Error during inference: {e}")
        inference.datamodule.teardown('predict')
        sys.exit(1)
