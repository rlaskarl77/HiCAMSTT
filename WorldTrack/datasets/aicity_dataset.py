import os
import numpy as np
import json
from torchvision.datasets import VisionDataset
import cv2
import re

class AiCity(VisionDataset):
    def __init__(self, root, num_cam=10):
        super().__init__(root)
        self.__name__ = 'AiCity'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1400, 800] 
        self.camera_numbers = self.get_camera_numbers()
        self.num_cam, self.num_frame = len(self.camera_numbers), 1600
        self.frame_step = 1

        self.worldcoord_from_worldgrid_mat = np.array([
            [0.025, 0.0, -10],  
            [0, 0.025, -15],  
            [0, 0, 1]         
        ])

        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_camera_numbers(self):
        camera_folders = [folder for folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets')))]
        camera_numbers = [int(re.search(r'\d+', folder).group()) for folder in camera_folders]
        return sorted(camera_numbers)

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = self.camera_numbers.index(int(camera_folder.split('_')[-1]))
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 800
        grid_y = pos // 800
        return np.array([grid_x, grid_y], dtype=int)

    def rq(self, M):
        Q, R = np.linalg.qr(np.flipud(M).transpose())
        R = np.flipud(R.transpose())
        R = np.fliplr(R)

        Q = Q.transpose()
        Q = np.flipud(Q)

        R = R * np.linalg.det(Q)
        Q = Q * np.linalg.det(Q)
        return R, Q

    def decompose_proj_numpy(self, P):
        K, R = self.rq(P[:3, :3])
        K = K / K[2,2]
        t = -np.linalg.inv(P[:3, :3]) @ P[:3, 3]
        return K, R, t

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        calibration_file = os.path.join(self.root, f'camera_{self.camera_numbers[camera_i]:04d}', 'calibration.json')
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        P = np.array(calibration_data["camera projection matrix"])

        K, R, t = self.decompose_proj_numpy(P)
        extrinsic_matrix = np.hstack([R, -R @ t[np.newaxis].T])
        return K, extrinsic_matrix