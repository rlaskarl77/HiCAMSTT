import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

class Factory(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'Factory'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1200, 300]  # H,W; N_row,N_col
        if '07' in self.root:
            self.worldgrid_shape = [1200, 300]
        elif '13' in self.root:
            self.worldgrid_shape = [1200, 500]
        self.num_cam, self.num_frame = 2, 365
        if '07' in self.root:
            self.num_frame = 365
        elif '13' in self.root:
            self.num_frame = 165
        self.frame_step = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.1, 0, -5], [0, 0.1, 90], [0, 0, 1]])
        if '07' in self.root:
            self.worldcoord_from_worldgrid_mat = np.array([[0.1, 0, -5], [0, 0.1, 90], [0, 0, 1]])
        elif '13' in self.root:
            self.worldcoord_from_worldgrid_mat = np.array([[0.1, 0, -5], [0, 0.1, 85], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos // 10000
        grid_y = pos % 10000
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        if camera_i == 0:
            cam = 'cam65'
            intrinsic_matrix = np.array([[982.761, 0.0, 988.18977], [0.0, 1128.76581, 510.34356], [0.0, 0.0, 1.0]])
            rvec = np.array([1.0980678550373892, 2.2184484772846442, -1.1604302766190677])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([-137.48520545941798, -65.94336472253306, 122.10626734402585]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        elif camera_i == 1:
            cam = 'cam74'
            intrinsic_matrix = np.array([[982.761, 0.0, 988.18977], [0.0, 1128.76581, 510.34356], [0.0, 0.0, 1.0]])
            rvec = np.array([0.9661684788441447, -2.2732914452355937, 1.2050721153835684])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([137.18635655042075, -65.3683602568311, 121.28470866951317]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
            
        return intrinsic_matrix, extrinsic_matrix


{
    "cam65": {
        "mtx": [
            [
                982.761,
                0.0,
                988.18977
            ],
            [
                0.0,
                1128.76581,
                510.34356
            ],
            [
                0.0,
                0.0,
                1.0
            ]
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
        ],
        "dist_coeffs": [
            -0.4577794,
            0.27236502,
            -0.00249496,
            -0.0012076,
            -0.0901687
        ]
    },
    "cam74": {
        "mtx": [
            [
                982.761,
                0.0,
                988.18977
            ],
            [
                0.0,
                1128.76581,
                510.34356
            ],
            [
                0.0,
                0.0,
                1.0
            ]
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
        ],
        "dist_coeffs": [
            -0.4577794,
            0.27236502,
            -0.00249496,
            -0.0012076,
            -0.0901687
        ]
    }
}
