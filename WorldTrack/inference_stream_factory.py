from datetime import datetime
import os.path as osp
import time
import torch
import lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
            learn_pose=True,
            id_pose_decompose=True,
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
        self.starter.record()
        
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
        
        # self.log_results(item['time'][0])
        
        try:
            self.draw_prediction(item, output, self.mota_now)
        except Exception as e:
            print(e)
        
        self.ender.record()
        torch.cuda.synchronize()
        
        ellapsed_time = self.starter.elapsed_time(self.ender)
        print(f'ellapsed_time: {ellapsed_time} ms')
        
            
            
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
    def __init__(self, model_configs, sources):
        self.models = {}
        self.datamodule = StreamFactoryDataModule(sources=sources)
        
        # Initialize models for each scene
        for scene_name, config in model_configs.items():
            checkpoint_path = config.pop('checkpoint_path')  # Remove checkpoint path from config
            model = WorldTrackModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                **config
            )
            model.eval()
            model.cuda()
            self.models[scene_name] = model

    def run_inference(self):
        self.datamodule.setup('predict')
        scene_loaders = self.datamodule.predict_dataloader()
        
        while True:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            all_results = []
            
            # Run inference on each scene
            for scene_name, loader in zip(self.models.keys(), scene_loaders):
                model = self.models[scene_name]
                
                try:
                    batch = next(iter(loader))
                    with torch.no_grad():
                        output = model.predict_step(batch, 0)
                        results = model.mota_now
                        all_results.extend(results)
                except StopIteration:
                    continue
                except Exception as e:
                    print(f"Error processing scene {scene_name}: {e}")
                    continue
            
            # Save combined results
            if all_results:
                hdc_data = self.convert_results_to_hdc_format(all_results, current_time)
                hdc_data.save_to_file(f"SNU_{current_time}_1.json")
            
            # Wait until next interval
            time.sleep(0.5)  # Adjust based on target FPS

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
                id=track_id,
                action=0,
                value=0,
                posx=x,
                posy=0.,
                posz=y,
                sizex=0,
                sizey=0,
                sizez=0,
                execution=0
            ))  

        return Data(
            time=time,
            camera=[Camera(camera_id="100", objects=object_list)]
        )

# Usage
model_configs = {
    'scene1': {
        'checkpoint_path': '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt',
        'resolution': (250, 4, 125),
        'bounds': (0, 500, 0, 1000, 0, 2),
        'scene_centroid': (0.0, 0.0, 0.0),
        'num_cameras': 2
    },
    'scene2': {
        'checkpoint_path': '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt',
        'resolution': (250, 4, 125),
        'bounds': (0, 500, 0, 1000, 0, 2),
        'scene_centroid': (0.0, 0.0, 0.0),
        'num_cameras': 2
    },
    'scene3': {
        'checkpoint_path': '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt',
        'resolution': (250, 4, 125),
        'bounds': (0, 500, 0, 1000, 0, 2),
        'scene_centroid': (0.0, 0.0, 0.0),
        'num_cameras': 2
    },
    'scene4': {
        'checkpoint_path': '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt',
        'resolution': (250, 4, 125),
        'bounds': (0, 500, 0, 1000, 0, 2),
        'scene_centroid': (0.0, 0.0, 0.0),
        'num_cameras': 2
    },
    'scene5': {
        'checkpoint_path': '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt',
        'resolution': (250, 4, 125),
        'bounds': (0, 500, 0, 1000, 0, 2),
        'scene_centroid': (0.0, 0.0, 0.0),
        'num_cameras': 2
    }
}

sources = [
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
    "rtsp://210.99.70.120:1935/live/cctv007.stream",
]

if __name__ == '__main__':
    import signal
    import sys

    torch.set_float32_matmul_precision('medium')
    
    inference = WorldTrackInference(model_configs, sources)
    
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
