import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset


class FactoryCam6878(VisionDataset):
    def __init__(self, root, num_frame):
        super().__init__(root)
        self.__name__ = 'Factory'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1000, 500]  
        self.num_cam, self.num_frame = 2, num_frame
        self.frame_step = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]]) # 68, 78
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
            cam = 'cam68'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                            [0.00000000e+00, 1128.76581, 510.34356],
                            [0, 0, 1]])
            rvec = np.array([2.0409456820227181e+00, 8.5610833576026513e-01, -5.6769163010387580e-01])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([-1.4730008344652514e+01, 2.9114570957977857e+00, 2.6220477687958013e+01]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        elif camera_i == 1: 
            cam = 'cam78'
            intrinsic_matrix = np.array([[982.76100, 0.00000000e+00, 988.18977],
                            [0.00000000e+00, 1128.76581, 510.34356],
                            [0, 0, 1]])
            rvec = np.array([1.8039614088785945e+00, -1.0075587724770085e+00, 5.7048706593582832e-01])
            rvec = cv2.Rodrigues(rvec)[0]
            tvec = np.array([2.5818068668951097e+00, 1.1432404299131896e+01, 1.0888543976918278e+01]).reshape(3, 1)
            extrinsic_matrix = np.hstack((rvec, tvec))
        return intrinsic_matrix, extrinsic_matrix
