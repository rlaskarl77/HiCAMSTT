import os
import json
import time
import datetime
import threading
from typing import Optional, Dict, List
from pathlib import Path

import cv2
import torch
import numpy as np
import pytorch_lightning as pl
import torch.multiprocessing as mp
from queue import Empty
from PIL import Image
import torchvision.transforms.functional as F

from utils import geom, basic, vox

cv2.setNumThreads(0)

# Constants and configs
CAMERA_PARAMS = {
    "cam63": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0378059978552201e+00, 2.1216798736914768e+00, -1.3374365036402904e+00
        ],
        "tvec": [
            -2.0317164441954540e+02, -7.4532298421297455e+01, 1.8774911045630381e+02
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam64": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9927930532158047e+00, 7.8779870669821117e-01, -6.9418659275046846e-01
        ],
        "tvec": [
            -1.5692849013275108e+02, 8.9076830587813419e+01, -7.3656232169674595e+01
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam65": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0980678550373892e+00, 2.2184484772846442e+00, -1.1604302766190677e+00
        ],
        "tvec": [
            -1.3748520545941798e+02, -6.5943364722533062e+01, 1.2210626734402585e+02
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam66": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8159877047382846e+00, 1.0241604870425753e+00, -5.4771387558579909e-01
        ],
        "tvec": [
            -9.1180789111459262e+01, 2.2453781104744184e+01, -2.0087448622042753e+01
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam68": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            2.0409456820227181e+00, 8.5610833576026513e-01, -5.6769163010387580e-01
        ],
        "tvec": [
            -1.4730008344652514e+01, 2.9114570957977857e+00, 2.6220477687958013e+01
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam72": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0711613881232815e+00, -2.0322384906462609e+00, 1.1785855921587649e+00
        ],
        "tvec": [
            2.3809508710896847e+02, -7.6804385502603324e+01, 1.3130606360817924e+02
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam73": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9856869409013398e+00, -7.3795425429308614e-01, 4.9885714288722405e-01
        ],
        "tvec": [
            1.3371296569471158e+02, 8.6270192534663138e+01, -1.0857030320299594e+02
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam74": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            9.6616847884414470e-01, -2.2732914452355937e+00, 1.2050721153835684e+00
        ],
        "tvec": [
            1.3718635655042075e+02, -6.5368360256831096e+01, 1.2128470866951317e+02
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam76": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9099458097732520e+00, -8.7512519018759150e-01, 5.4748044457257483e-01
        ],
        "tvec": [
            7.2757984828162279e+01, 4.0244222510388106e+01, -4.0282195276009467e+01
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    },
    "cam78": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8039614088785945e+00, -1.0075587724770085e+00, 5.7048706593582832e-01
        ],
        "tvec": [
            2.5818068668951097e+00, 1.1432404299131896e+01, 1.0888543976918278e+01
        ],
        "mtx_orig": [
            [1.17969312e+03, 0.0e+0, 9.53868716e+02],
            [0.0e+0, 1.2476522e+03,5.27779005e+02],
            [0.0e+0, 0.0e+0, 1.0e+0]
        ],
        "dist_coeff": [-0.4577794, 0.27236502, -0.00249496, -0.0012076, -0.0901687]
    }
}

SCENE_CONFIGS = {
    "scene1": {
        "cameras": (63, 72),
        "worldcoord_from_worldgrid_mat": [[0.05, 0, 0], [0, 0.05, 240], [0, 0, 1]]
    },
    "scene2": {
        "cameras": (64, 73),
        "worldcoord_from_worldgrid_mat": [[0.05, 0, 0], [0, 0.05, 185], [0, 0, 1]]
    },
    "scene3": {
        "cameras": (65, 74),
        "worldcoord_from_worldgrid_mat": [[0.05, 0, 0], [0, 0.05, 150], [0, 0, 1]]
    },
    "scene4": {
        "cameras": (66, 76),
        "worldcoord_from_worldgrid_mat": [[0.05, 0, 0], [0, 0.05, 85], [0, 0, 1]]
    },
    "scene5": {
        "cameras": (68, 78),
        "worldcoord_from_worldgrid_mat": [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]]
    }
}

mp.set_start_method('spawn', force=True)
class StreamManager:
    """Manages multiple RTSP streams with synchronized frame access"""
    
    def __init__(self, sources: List[str], target_fps: float = 2.0, verbose=False, init_event=None):
        self.sources = sources
        self.n_streams = len(sources)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.verbose = verbose  # verbose 옵션 저장
        self.caps = []
        self.running = True
        self.sync_lock = threading.Lock()
        # 단일 프레임 버퍼로 변경
        self.latest_frames = [None] * self.n_streams
        self.last_frame_time = [0.0] * self.n_streams
        self.threads = []
        # self.setup_streams()
        self.frame_times = []  # Add timing tracker
        
        self.init_event = init_event

    def setup_streams(self):
        """Initialize video captures and start reader threads"""
        for i, source in enumerate(self.sources):
            if self.verbose:
                print(f"[StreamManager] Opening stream {i}: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[StreamManager] Failed to open stream {i}: {source}")
                raise ConnectionError(f"Failed to open stream {i}: {source}")
            self.caps.append(cap)
            thread = threading.Thread(
                target=self._read_stream,
                args=(i, cap),
                daemon=True
            )
            self.threads.append(thread)
            thread.start()
            if self.verbose:
                print(f"[StreamManager] Stream {i} started.")
        if self.init_event is not None:
            self.init_event.set()

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
            self.frame_times.append(current_time)
            
            # Calculate actual FPS
            if len(self.frame_times) > 10:
                time_diff = self.frame_times[-1] - self.frame_times[-10]
                actual_fps = 10 / time_diff
                print(f"Actual FPS: {actual_fps:.2f}")
                self.frame_times = self.frame_times[-10:]  # Keep last 10
                
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
            verbose: bool = False
    ):
        self.scene_name = scene_name
        self.scene_config = SCENE_CONFIGS[scene_name]
        self.stream_manager = stream_manager
        self.stream_indices = stream_indices
        
        self.verbose = verbose  # verbose 옵션 저장
        
        # Parameters
        self.num_cam = 2
        self.img_shape = (1080, 1920, 3)
        self.worldgrid_shape = [bounds[3], bounds[1]]
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
    
    def undistort_image(self, img, cam_id):
        
        # change 1280 x 720 to 1920 x 1080
        img = img.resize((1920, 1080), Image.NEAREST)
        
        # Convert PIL Image to numpy array
        img = np.array(img)
        
        # convert to cv2 BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        # Get camera parameters
        mtx_orig = np.array(CAMERA_PARAMS[f'cam{cam_id}']['mtx_orig'])
        dist_coeff = np.array(CAMERA_PARAMS[f'cam{cam_id}']['dist_coeff'])
        
        # Get image dimensions
        w, h = img.shape[1], img.shape[0]
        alpha = 0.25
        
        # Undistort
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_orig, dist_coeff, (w, h), alpha, (w, h))
        dst = cv2.undistort(img, mtx_orig, dist_coeff, None, newcameramtx)
        
        
        # Convert back to RGB
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        dst = Image.fromarray(dst)
        
        # change back to 1280 x 720
        dst = dst.resize((1280, 720), Image.NEAREST)
        
        return dst

    def get_image_data(self, images, cameras):
        imgs, intrins, extrins = [], [], []
        
        fH, fW = self.data_aug_conf['final_dim']
        
        for img, cam in zip(images, cameras):
            # Convert BGR to RGB and to PIL Image
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Undistort image (now handles PIL Image correctly)
            img = self.undistort_image(img, self.scene_config['cameras'][cam])
            
            # W, H = img.size
            W, H = 1920, 1080
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
            target_fps: float = 2.0,
            verbose=False):
        super().__init__()
        self.sources = sources
        self.resolution = resolution
        self.bounds = bounds
        self.final_dim = final_dim
        self.target_fps = target_fps
        self.verbose = verbose  # verbose 옵션 저장
        self.processes = []
        self.queues = {}
        self.init_events = {}

    def setup(self, stage: Optional[str] = None):
        if stage == 'predict':
            # Create queues for each scene
            for scene_name in SCENE_CONFIGS.keys():
                self.queues[scene_name] = mp.Queue(maxsize=2)
            
            # Start processes for each scene
            for scene_idx, (scene_name, config) in enumerate(SCENE_CONFIGS.items()):
                stream_indices = [scene_idx*2, scene_idx*2+1]
                process_args = {
                    'scene_name': scene_name,
                    'stream_indices': stream_indices,
                    'sources': self.sources,
                    'resolution': self.resolution,
                    'bounds': self.bounds,
                    'final_dim': self.final_dim,
                    'target_fps': self.target_fps,
                    'verbose': self.verbose  # verbose 옵션 전달
                }
                
                init_event = mp.Event()
                self.init_events[scene_name] = init_event
                
                p = mp.Process(
                    target=self._process_scene,
                    args=(process_args, self.queues[scene_name], init_event),
                    daemon=False  # daemon=False로 설정
                )
                p.start()
                if self.verbose:
                    print(f"Started process {p.pid} for {scene_name}")
                self.processes.append(p)

    @staticmethod
    def _process_scene(args, queue, init_event=None):
        verbose = args.get('verbose', False)
        try:
            if verbose:
                print(f"[{args['scene_name']}] Starting process.")
            selected_sources = [args['sources'][i] for i in args['stream_indices']]
            if verbose:
                print(f"[{args['scene_name']}] Selected sources: {selected_sources}")
            stream_manager = StreamManager(selected_sources, args['target_fps'], verbose=verbose,
                                           init_event=init_event)
            stream_manager.setup_streams()
            stream_indices = list(range(len(selected_sources)))
            stream = LoadStreams(
                scene_name=args['scene_name'],
                stream_manager=stream_manager,
                stream_indices=stream_indices,
                resolution=args['resolution'],
                bounds=args['bounds'],
                final_dim=args['final_dim'],
                verbose=verbose
            )
            if verbose:
                print(f"[{args['scene_name']}] Stream initialized.")
            while True:
                try:
                    data = next(stream)
                    if verbose:
                        print(f"[{args['scene_name']}] Frame received.")
                    # 데이터 변환 및 큐에 삽입
                    data = {
                        'img': data['img'],
                        'intrinsic': data['intrinsic'],
                        'extrinsic': data['extrinsic'],
                        'ref_T_global': data['ref_T_global'],
                        'scene_name': data['scene_name'],
                        'camera_ids': data['camera_ids'],
                        'time': data['time'],
                        'sequence_num': data['sequence_num']
                    }
                    queue.put(data)
                except KeyboardInterrupt:
                    if verbose:
                        print(f"[{args['scene_name']}] Process interrupted by user.")
                    break
                except Exception as e:
                    if verbose:
                        print(f"Error in process {args['scene_name']}: {e}")
                    continue
        except Exception as e:
            if verbose:
                print(f"[{args['scene_name']}] Process failed: {e}")

    def predict_dataloader(self):
        return MultiProcessDataLoader(self.queues, self.target_fps)

    def teardown(self, stage: Optional[str] = None):
        for p in self.processes:
            p.terminate()
        for p in self.processes:
            p.join()
        for q in self.queues.values():
            q.close()

class MultiProcessDataLoader:
    def __init__(self, queues, target_fps, verbose=False):
        self.queues = queues
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()
        
        self.verbose = verbose  # verbose 옵션 저장
        if self.verbose:
            print(f"Initialized loader with {len(queues)} queues")

    def __iter__(self):
        return self

    def __next__(self):
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        try:
            scene_data = {}
            # Get data from all scenes with longer timeout
            for scene_name, queue in self.queues.items():
                if self.verbose:
                    print(f"Waiting for data from {scene_name}...")
                try:
                    scene_data[scene_name] = queue.get(timeout=1.0)  # Increased timeout
                    if self.verbose:
                        print(f"Received data from {scene_name}")
                except Empty:
                    if self.verbose:
                        print(f"Timeout waiting for {scene_name}")
                    continue
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting data from {scene_name}: {e}")
                    continue
            
            if not scene_data:
                print("No data received from any scene")
                time.sleep(0.1)
                return self.__next__()
            
            self.last_frame_time = time.time()
            if self.verbose:
                print(f"Returning data for {len(scene_data)} scenes")
            return scene_data
            
        except Exception as e:
            print(f"Error in dataloader: {e}")
            raise StopIteration

def test_dataloader_fps(datamodule, verbose=False):
    # 스트림 초기화 대기
    if verbose:
        print("Waiting for all streams to initialize...")
    for scene_name, event in datamodule.init_events.items():
        event.wait()
        if verbose:
            print(f"{scene_name} initialized.")
    if verbose:
        print("All streams initialized. Starting data loading.")
        
    frame_times = []
    frame_count = 0
    start_time = time.time()
    loader = datamodule.predict_dataloader()
    
    try:
        if verbose:
            print("Starting FPS test...")
        while time.time() - start_time < 20:  # Run for 20 seconds
            try:
                scene_data = next(iter(loader))
                current_time = time.time()
                frame_times.append(current_time)
                frame_count += 1
                
                # Wait for at least 2 frames before calculating FPS
                if len(frame_times) > 10:
                    time_diff = frame_times[-1] - frame_times[-10]
                    if time_diff > 0:  # Prevent division by zero
                        current_fps = 10 / time_diff
                        print(f"Current FPS: {current_fps:.2f}")
                    frame_times = frame_times[-10:]
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
            except StopIteration:
                print("No more frames available")
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Cleanup
        datamodule.teardown('predict')
        
        if frame_times and len(frame_times) > 1:
            total_time = frame_times[-1] - frame_times[0]
            if total_time > 0:
                avg_fps = len(frame_times) / total_time
                print(f"\nTest Summary:")
                print(f"Total frames: {frame_count}")
                print(f"Average FPS: {avg_fps:.2f}")
                
    # Check process status after the test
    for p in datamodule.processes:
        if not p.is_alive():
            print(f"Process {p.pid} ({p.name}) has stopped.")

def test_stream_connection(sources):
    for idx, source in enumerate(sources):
        print(f"Testing connection to stream {idx}: {source}")
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully read frame from stream {idx}")
            else:
                print(f"Failed to read frame from stream {idx}")
            cap.release()
        else:
            print(f"Failed to open stream {idx}")

def main(verbose=False):
    test_sources = [
        "rtsp://210.99.70.120:1935/live/cctv007.stream",
        "rtsp://210.99.70.120:1935/live/cctv008.stream",
    ] * 5

    datamodule = None

    try:
        if verbose:
            print("Initializing datamodule...")
        datamodule = StreamFactoryDataModule(sources=test_sources, verbose=verbose)

        if verbose:
            print("Setting up datamodule...")
        datamodule.setup('predict')

        if verbose:
            print("Starting FPS test...")
        test_dataloader_fps(datamodule, verbose=verbose)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if verbose:
            print("\nCleaning up resources...")
        if datamodule is not None:
            try:
                datamodule.teardown('predict')
                for p in datamodule.processes:
                    if p.is_alive():
                        if verbose:
                            print(f"Terminating process {p.pid}")
                        p.terminate()
                        p.join(timeout=1.0)
            except Exception as e:
                print(f"Error during cleanup: {e}")


# if __name__ == '__main__':
#     # Try to set start method only if not already set
#     try:
#         mp.get_start_method()
#     except RuntimeError:
#         mp.set_start_method('spawn', force=True)
if __name__ == '__main__':
    main(verbose=False)  # 필요에 따라 True 또는 False로 설정