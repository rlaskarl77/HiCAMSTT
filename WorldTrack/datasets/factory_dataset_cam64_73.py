import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset


class FactoryCam6473(VisionDataset):
    def __init__(self, root, num_frame):
        super().__init__(root)
        self.__name__ = 'Factory'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1000, 500]  
        self.num_cam, self.num_frame = 2, num_frame
        self.num_cam = 2
        self.frame_step = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.05, 0, 0], [0, 0.05, 185], [0, 0, 1]]) # 64, 73
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
            cam = 'cam64'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                        [0.00000000e+00, 1128.76581, 510.34356],
                        [0, 0, 1]])
            
            rvec = np.array([1.9927930532158047e+00, 7.8779870669821117e-01, -6.9418659275046846e-01])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([-1.5692849013275108e+02, 8.9076830587813419e+01, -7.3656232169674595e+01]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        elif camera_i == 1:  
            cam = 'cam73'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                        [0.00000000e+00, 1128.76581, 510.34356],
                        [0, 0, 1]])
            
            rvec = np.array([1.9856869409013398e+00, -7.3795425429308614e-01, 4.9885714288722405e-01])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([1.3371296569471158e+02, 8.6270192534663138e+01, -1.0857030320299594e+02]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        return intrinsic_matrix, extrinsic_matrix
