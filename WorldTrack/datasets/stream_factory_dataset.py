import os
import json
from operator import itemgetter
from pathlib import Path
from urllib.parse import urlparse
import threading
from threading import Thread
import time
import math
import re
import datetime
import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Dict, List
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image

from utils import geom, basic, vox

# Camera parameters
CAMERA_PARAMS = {
    "cam63": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.03780599785522,
            2.121679873691477,
            -1.3374365036402904
        ],
        "tvec": [
            -203.1716444195454,
            -74.53229842129745,
            187.7491104563038
        ]
    },
    "cam64": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9927930532158047,
            0.7877987066982112,
            -0.6941865927504685
        ],
        "tvec": [
            -156.92849013275108,
            89.07683058781342,
            -73.6562321696746
        ]
    },
    "cam65": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0980678550373892,
            2.2184484772846442,
            -1.1604302766190677
        ],
        "tvec": [
            -137.48520545941798,
            -65.94336472253306,
            122.10626734402585
        ]
    },
    "cam66": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8159877047382846,
            1.0241604870425753,
            -0.5477138755857991
        ],
        "tvec": [
            -91.18078911145926,
            22.453781104744184,
            -20.087448622042753
        ]
    },
    "cam68": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            2.040945682022718,
            0.8561083357602651,
            -0.5676916301038758
        ],
        "tvec": [
            -14.730008344652514,
            2.9114570957977857,
            26.220477687958013
        ]
    },
    "cam72": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0711613881232815,
            -2.032238490646261,
            1.178585592158765
        ],
        "tvec": [
            238.09508710896847,
            -76.80438550260332,
            131.30606360817924
        ]
    },
    "cam73": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9856869409013398,
            -0.7379542542930861,
            0.49885714288722405
        ],
        "tvec": [
            133.71296569471158,
            86.27019253466314,
            -108.57030320299594
        ]
    },
    "cam74": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            0.9661684788441447,
            -2.2732914452355937,
            1.2050721153835684
        ],
        "tvec": [
            137.18635655042075,
            -65.3683602568311,
            121.28470866951317
        ]
    },
    "cam76": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.909945809773252,
            -0.8751251901875915,
            0.5474804445725748
        ],
        "tvec": [
            72.75798482816228,
            40.244222510388106,
            -40.28219527600947
        ]
    },
    "cam78": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8039614088785945,
            -1.0075587724770085,
            0.5704870659358283
        ],
        "tvec": [
            2.5818068668951097,
            11.432404299131896,
            10.888543976918278
        ]
    }
}

# Scene configurations
SCENE_CONFIGS = {
    "scene1": {
        "cameras": (63, 72),
        "worldcoord_from_worldgrid_mat": [[0.1, 0, 0], [0, 0.1, 240], [0, 0, 1]]
    },
    "scene2": {
        "cameras": (64, 73),
        "worldcoord_from_worldgrid_mat": [[0.1, 0, 0], [0, 0.1, 190], [0, 0, 1]]
    },
    "scene3": {
        "cameras": (65, 74),
        "worldcoord_from_worldgrid_mat": [[0.1, 0, 0], [0, 0.1, 150], [0, 0, 1]]
    },
    "scene4": {
        "cameras": (66, 76),
        "worldcoord_from_worldgrid_mat": [[0.1, 0, 0], [0, 0.1, 85], [0, 0, 1]]
    },
    "scene5": {
        "cameras": (68, 78),
        "worldcoord_from_worldgrid_mat": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]
    }
}

class StreamManager:
    """Manages multiple RTSP streams with synchronized frame access"""
    
    def __init__(self, sources: List[str], target_fps: float = 2.0):
        self.sources = sources
        self.n_streams = len(sources)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.caps = []
        self.running = True
        self.sync_lock = threading.Lock()
        # 단일 프레임 버퍼로 변경
        self.latest_frames = [None] * self.n_streams
        self.last_frame_time = [0.0] * self.n_streams
        self.threads = []
        self.setup_streams()

    def setup_streams(self):
        """Initialize video captures and start reader threads"""
        for i, source in enumerate(self.sources):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ConnectionError(f"Failed to open stream {i}: {source}")
            self.caps.append(cap)
            thread = threading.Thread(
                target=self._read_stream,
                args=(i, cap),
                daemon=True
            )
            self.threads.append(thread)
            thread.start()

    def _read_stream(self, stream_id: int, cap: cv2.VideoCapture):
        """Reader thread for each stream with FPS control"""
        while self.running and cap.isOpened():
            current_time = time.time()
            
            if current_time - self.last_frame_time[stream_id] < self.frame_interval:
                success = cap.grab()
                if not success:
                    continue
                time.sleep(0.001)
                continue
                
            success, frame = cap.read()
            if not success:
                continue
                
            with self.sync_lock:
                self.latest_frames[stream_id] = frame
                self.last_frame_time[stream_id] = current_time

    def get_synchronized_frames(self, stream_indices: List[int]) -> List[np.ndarray]:
        """Get synchronized frames for specified stream indices"""
        frames = []
        with self.sync_lock:
            current_time = time.time()
            # 모든 스트림의 프레임이 준비되었는지 확인
            for idx in stream_indices:
                if self.latest_frames[idx] is None:
                    return None
                # 프레임이 너무 오래되었는지 확인 (1초 이상)
                if current_time - self.last_frame_time[idx] > 1.0:
                    return None
            frames = [self.latest_frames[idx].copy() for idx in stream_indices]
        return frames

    def close(self):
        """Release all resources"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        for cap in self.caps:
            cap.release()

class LoadStreams:
    def __init__(
            self, 
            scene_name: str,
            stream_manager: StreamManager,
            stream_indices: List[int],
            resolution=(160, 4, 250),
            bounds=(0, 500, 0, 1000, 0, 2),
            final_dim: tuple = (720, 1280),
    ):
        self.scene_name = scene_name
        self.scene_config = SCENE_CONFIGS[scene_name]
        self.stream_manager = stream_manager
        self.stream_indices = stream_indices
        
        # Parameters
        self.num_cam = 2
        self.img_shape = (1080, 1920, 3)
        self.worldgrid_shape = resolution[0::2]
        self.resolution = resolution
        self.bounds = bounds
        self.data_aug_conf = {'final_dim': final_dim}
        
        self.count = 0
        self.setup_calibration()

    def setup_calibration(self):
        """Setup camera calibration parameters"""
        cam_ids = self.scene_config['cameras']
        self.intrinsic_matrices = []
        self.extrinsic_matrices = []
        
        for cam_id in cam_ids:
            cam_key = f'cam{cam_id}'
            cam_params = CAMERA_PARAMS[cam_key]
            
            self.intrinsic_matrices.append(np.array(cam_params['mtx']))
            
            # Extrinsic matrix (4x4)
            rvec = np.array(cam_params['rvec'])
            tvec = np.array(cam_params['tvec'])
            R, _ = cv2.Rodrigues(rvec)
            
            # Create 4x4 homogeneous transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec
            self.extrinsic_matrices.append(T)

    def __repr__(self):
        return f"LoadStreams(scene={self.scene_name}, cameras={self.scene_config['cameras']})"

    def __iter__(self):
        self.count = 0
        return self

    def get_image_data(self, images, cameras):
        imgs, intrins, extrins = [], [], []
        
        fH, fW = self.data_aug_conf['final_dim']
        
        for img, cam in zip(images, cameras):
            # Convert BGR to RGB and to PIL Image
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            W, H = img.size

            sx = fW / float(W)
            sy = fH / float(H)

            # Convert matrices to torch tensors
            intrin = torch.from_numpy(self.intrinsic_matrices[cam]).float()
            extrin = torch.from_numpy(self.extrinsic_matrices[cam]).float()

            # Scale intrinsics using geom utility
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)
            
            # Convert to camera transformation matrix
            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))
            pix_T_cam = geom.merge_intrinsics(fx, fy, x0, y0)
            intrin = pix_T_cam.squeeze(0)  # 4,4
            
            # Resize and convert image
            img = img.resize((fW, fH), Image.NEAREST)
            # Create writable numpy array
            img = np.array(img, dtype=np.uint8).copy()
            
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins)

    def __next__(self):
        try:
            self.count += 1
            
            images = self.stream_manager.get_synchronized_frames(self.stream_indices)
            if images is None:
                time.sleep(0.01)
                return self.__next__()
                
            cameras = list(range(self.num_cam))
            imgs, intrins, extrins = self.get_image_data(images, cameras)

            worldcoord_from_worldgrid = torch.eye(4)
            worldcoord_from_worldgrid2d = \
                torch.tensor(self.scene_config["worldcoord_from_worldgrid_mat"], dtype=torch.float32)
            worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
            worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
            worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)
            
            imgs = imgs.unsqueeze(0)
            intrins = intrins.unsqueeze(0)
            extrins = extrins.unsqueeze(0)
            worldgrid_T_worldcoord = worldgrid_T_worldcoord.unsqueeze(0)
            
            return {
                'img': imgs,
                'intrinsic': intrins,
                'extrinsic': extrins,
                'ref_T_global': worldgrid_T_worldcoord,  # 4,4
                'scene_name': self.scene_name,
                'camera_ids': self.scene_config['cameras'],
                'time': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]],
                'sequence_num': [int(0)],
            }
        except Exception as e:
            print(f"Error in LoadStreams.__next__: {e}")
            raise StopIteration

class StreamFactoryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            sources: List[str],
            resolution=(250, 4, 125),
            bounds=(0, 500, 0, 1000, 0, 2),
            final_dim: tuple = (720, 1280),
            target_fps: float = 2.0
    ):
        super().__init__()
        self.sources = sources
        self.resolution = resolution
        self.bounds = bounds
        self.final_dim = final_dim
        self.target_fps = target_fps
        
        self.stream_manager = None
        self.scene_streams = {}

    def setup(self, stage: Optional[str] = None):
        if stage == 'predict':
            self.stream_manager = StreamManager(
                sources=self.sources,
                target_fps=self.target_fps
            )
            
            for scene_idx, (scene_name, config) in enumerate(SCENE_CONFIGS.items()):
                stream_indices = [scene_idx*2, scene_idx*2+1]
                self.scene_streams[scene_name] = LoadStreams(
                    scene_name=scene_name,
                    stream_manager=self.stream_manager,
                    stream_indices=stream_indices,
                    resolution=self.resolution,
                    bounds=self.bounds,
                    final_dim=self.final_dim,
                )

    def predict_dataloader(self):
        return list(self.scene_streams.values())

    def teardown(self, stage: Optional[str] = None):
        if self.stream_manager:
            self.stream_manager.close()

if __name__ == "__main__":
    '''
    Test the code by running the following commands:
    cd WorldTrack
    python -m datasets.stream_factory_dataset
    '''
    # Usage
    
    test_sources = [
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

    datamodule = StreamFactoryDataModule(sources=test_sources)

    datamodule.setup(stage='predict')

    # 수정된 테스트 코드
    try:
        for stream in datamodule.predict_dataloader():
            print(f"Processing stream: {stream}")
            for data in stream:
                print(f"Scene: {data['scene_name']}, Time: {data['time']}")
                print(data["img"].shape, data["intrinsic"].shape, data["extrinsic"].shape, data["ref_T_global"].shape)
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        datamodule.teardown('predict')
        