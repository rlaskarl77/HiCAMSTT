import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

# intrinsic_camera_matrix_filenames = ['intr_cam1_0.xml', 'intr_cam2_0.xml', 'intr_cam3_0.xml']
# intrinsic_camera_matrix_filenames = ['reshaped_intr_cam1.xml', 'reshaped_intr_cam2.xml', 'reshaped_intr_cam3.xml']
# intrinsic_camera_matrix_filenames = ['reshaped_intr_cam1.xml', 'intr_cam2_0.xml', 'intr_cam3_0.xml']
# extrinsic_camera_matrix_filenames = ['extr_cam1_0.xml', 'extr_cam2_0.xml', 'extr_cam3_0.xml']

intrinsic_camera_matrix_filenames = ['intr_cam1.xml', 'intr_cam2.xml', 'intr_cam3.xml']
extrinsic_camera_matrix_filenames = ['extr_cam1.xml', 'extr_cam2.xml', 'extr_cam3.xml']


class HDC(VisionDataset):
    def __init__(self, root='/131_data/datasets/HiCAMS/20240702'):
        super().__init__(root)
        
        self.__name__ = 'HDC'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [900, 900]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 3, 806
        self.frame_step = 1
        
        self.worldcoord_from_worldgrid_mat = np.array([[1, 0, -450], [0, 1, -450], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'images'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'images', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'images', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos // 900
        grid_y = pos % 900
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
    
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsics')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()
        
        
        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsics')
        extrinsic_params_file = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                             extrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        
        rvec = extrinsic_params_file.getNode('rvec').mat().squeeze()
        tvec = extrinsic_params_file.getNode('tvec').mat().squeeze()
        extrinsic_params_file.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix
