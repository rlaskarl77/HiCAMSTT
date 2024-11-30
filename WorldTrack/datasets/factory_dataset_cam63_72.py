import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset


class FactoryCam6372(VisionDataset):
    def __init__(self, root, num_frame):
        super().__init__(root)
        self.__name__ = 'Factory'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1000, 500]  
        self.num_cam, self.num_frame = 2, num_frame
        self.num_cam = 2
        self.frame_step = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.05, 0, 0], [0, 0.05, 240], [0, 0, 1]]) # 63, 72
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1])
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    def get_worldgrid_from_pos(self, pos):
        grid_x = int(pos) %  500
        grid_y = int(pos) // 500
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        if camera_i == 0:
            cam = 'cam63'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                            [0.00000000e+00, 1128.76581, 510.34356],
                            [0, 0, 1]])
            
            rvec = np.array([1.0378059978552201e+00, 2.1216798736914768e+00, -1.3374365036402904e+00])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec =  np.array([-2.0317164441954540e+02, -7.4532298421297455e+01, 1.8774911045630381e+02]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        elif camera_i == 1:  
            cam = 'cam72'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                            [0.00000000e+00, 1128.76581, 510.34356],
                            [0, 0, 1]])
            rvec = np.array([1.0711613881232815e+00, -2.0322384906462609e+00, 1.1785855921587649e+00])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([2.3809508710896847e+02, -7.6804385502603324e+01, 1.3130606360817924e+02]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        return intrinsic_matrix, extrinsic_matrix
