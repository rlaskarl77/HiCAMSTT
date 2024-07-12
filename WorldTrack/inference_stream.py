import os.path as osp
import torch
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

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
            max_detections=60,
            conf_threshold=0.5,
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
        
        self.geometric_loss_fn = torch.nn.functional.smooth_l1_loss

        # Temporal cache
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 32
        self.temporal_cache_frames = [""] * self.max_cache
        self.temporal_cache = None

        # Test
        self.moda_gt_list, self.moda_pred_list = [], []
        self.mota_gt_list, self.mota_pred_list = [], []
        self.mota_seq_gt_list, self.mota_seq_pred_list = [], []
        
        
        self.moda_now, self.mota_now = [], []
        
        self.frame = 0
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold)

        # Model
        num_cameras = None if num_cameras == 0 else num_cameras
        if model_name == 'segnet':
            self.model = Segnet(self.Y, self.Z, self.X, num_cameras=num_cameras, feat2d_dim=feat2d_dim,
                                encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign)
        elif model_name == 'liftnet':
            self.model = Liftnet(self.Y, self.Z, self.X, encoder_type=self.encoder_name, feat2d_dim=feat2d_dim,
                                 DMIN=self.DMIN, DMAX=self.DMAX, D=self.D, num_classes=num_classes, z_sign=z_sign,
                                 num_cameras=num_cameras)
        elif model_name == 'bevformer':
            self.model = Bevformernet(self.Y, self.Z, self.X, feat2d_dim=feat2d_dim,
                                      encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign)
        elif model_name == 'mvdet':
            self.model = MVDet(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                               num_cameras=num_cameras, num_classes=num_classes)
        elif model_name == 'mvdetr':
            self.model = MVDetr(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                                num_cameras=num_cameras, num_classes=num_classes, feat2d_dim=feat2d_dim)
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

    def load_cache(self, frames):
        idx = []
        frames = [frames] if isinstance(frames, str) else frames
        for frame in frames:
            try:
                idx.append(self.temporal_cache_frames.index(frame))
            except ValueError:
                return None
        if len(idx) != len(frames):
            return None
        else:
            return self.temporal_cache[idx]

    def store_cache(self, times, bev_feat):
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape, device=bev_feat.device, dtype=bev_feat.dtype)

        for time, feat in zip(times, bev_feat):
            try:
                i = next(i for i, s in enumerate(self.temporal_cache_frames) if not s)
            except StopIteration:
                i = self.temporal_cache_frames.index(min(self.temporal_cache_frames))
            self.temporal_cache[i] = feat
            self.temporal_cache_frames[i] = time

    def predict_step(self, batch, batch_idx):
        
        self.starter.record()
        
        item = batch
        output = self(item)

        # ref_T_global = item['ref_T_global']
        # global_T_ref = torch.inverse(ref_T_global)

        # output on bev plane
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
        for frame, xy, score in zip(item['time'], ref_xy, scores_e):
            valid = score > self.conf_threshold
            frame = int(frame)
            self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])
            self.moda_now.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])

        # tracking
        for seq_num, frame, bev_det, bev_prev, score, in (
                zip(item['sequence_num'], item['time'], ref_xy.cpu(), ref_xy_prev.cpu(),
                    scores_e.cpu())):
            frame = int(frame)
            output_stracks = self.test_tracker.update(bev_det, bev_prev, score)
            self.mota_pred_list.extend([[seq_num, frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                                        + s.xy.tolist() + [-1]
                                        for s in output_stracks])
            
            self.mota_now.extend([[seq_num, frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                                        + s.xy.tolist() + [-1]
                                        for s in output_stracks])
        
        self.log_results(item['time'][0])
        
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
        hdc_data.save_to_file("result.json")

         
    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        np.savetxt(pred_path, np.array(self.moda_now), '%f')

        pred_path = osp.join(log_dir, 'mota_pred.txt')
        np.savetxt(pred_path, np.array(self.mota_now), '%f', delimiter=',')

    def convert_mota_to_hdc_format(self, mota_pred_list, time):
        data = np.asarray(mota_pred_list)
        data = data[:, (1, 2, 8, 9)]
        object_list = []
        for frame, track_id, x, y in data:
            object_list.append(ObjectType(
                    type=0,
                    id=track_id,
                    action=0,
                    value=0,
                    posx=x,
                    posy=y,
                    posz=0.0,
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
        
if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")


    cli = MyLightningCLI(WorldTrackModel)
