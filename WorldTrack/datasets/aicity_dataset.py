import os
import numpy as np
import json
from torchvision.datasets import VisionDataset
import cv2

class AiCity(VisionDataset):
    def __init__(self, root, num_cam=10):
        super().__init__(root)
        self.__name__ = 'AiCity'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1400, 800] 
        # self.num_cam, self.num_frame = 1, 23994
        self.num_cam, self.num_frame = 10, 50
        self.frame_step = 1
        
        self.worldcoord_from_worldgrid_mat = np.array([
            [0, 0.025, -10],  
            [0.025, 0, -15],  
            [0, 0, 1]         
        ])
        
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder.split('_')[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_y = pos % 1400
        grid_x = pos // 1400
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        calibration_file = os.path.join(self.root, f'camera_{camera_i + 1:04d}', 'calibration.json')
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        P = np.array(calibration_data["camera projection matrix"])

        p1 = P[:, 0]
        p2 = P[:, 1]
        p3 = P[:, 2]
        p4 = P[:, 3]
        M = np.column_stack((p1, p2, p3))

        X = np.linalg.det(np.column_stack((p2, p3, p4)))
        Y = -np.linalg.det(np.column_stack((p1, p3, p4)))
        Z = np.linalg.det(np.column_stack((p1, p2, p4)))
        T = -np.linalg.det(np.column_stack((p1, p2, p3)))

        Pc = np.array([X, Y, Z, T])
        Pc = Pc / Pc[3]  
        Pc = Pc[:3] 

        result = cv2.RQDecomp3x3(M)
        K, R = result[1], result[2]

        T_matrix = np.diag(np.sign(np.diag(K)))
        K = K @ T_matrix
        R = T_matrix @ R

        extrinsic_matrix = np.hstack((R, -R @ Pc.reshape(3, 1)))
        return K, extrinsic_matrix


# def test_get_intrinsic_extrinsic_matrix():
#     root = "/data/aicity/scene_001/"
#     camera_i = 0  

#     calibration_file = os.path.join(root, f'camera_{camera_i + 1:04d}', 'calibration.json')
#     with open(calibration_file, 'r') as f:
#         calibration_data = json.load(f)
#     print(calibration_file)
#     P = np.array(calibration_data["camera projection matrix"])

#     dataset = AiCity(root)
#     K, extrinsic = dataset.get_intrinsic_extrinsic_matrix(camera_i)
#     print("extrinsic: ", extrinsic.shape)
#     print("Intrinsic matrix K:\n", K)
#     print("Extrinsic matrix [R|T]:\n", extrinsic)


# # 테스트 실행
# test_get_intrinsic_extrinsic_matrix()
