import os
import os.path as osp
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from kornia.geometry.transform.imgwarp import warp_perspective
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


intrinsic_camera_matrix_filenames = ['intr_cam1.xml', 'intr_cam2.xml', 'intr_cam3.xml']
extrinsic_camera_matrix_filenames = ['extr_cam1.xml', 'extr_cam2.xml', 'extr_cam3.xml']

DATA_PATH = '/131_data/datasets/HiCAMS/20240110'
SAVE_PATH = '/home/namgi/TrackTacular/visualization/hdc/demo'
# SAVE_PATH = '/home/namgi/TrackTacular/visualization/hdc/demo02'
PRED_FILE = 'mota_pred.txt'

font_size = 50
font = ImageFont.truetype("Arial.ttf", font_size)
font_color = (255, 64, 0)

def get_intrinsic_extrinsic_matrix(camera_i):
    intrinsic_camera_path = os.path.join(DATA_PATH, 'calibrations', 'intrinsics')
    fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                intrinsic_camera_matrix_filenames[camera_i]),
                                    flags=cv2.FILE_STORAGE_READ)
    intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
    fp_calibration.release()

    extrinsic_camera_path = os.path.join(DATA_PATH, 'calibrations', 'extrinsics')
    fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                extrinsic_camera_matrix_filenames[camera_i]),
                                    flags=cv2.FILE_STORAGE_READ)
    rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
    fp_calibration.release()

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

    return intrinsic_matrix, extrinsic_matrix
    
def project_2d_points(intrinsic_mat, extrinsic_mat, input_points, x_offset, y_offset, z_offset):
    vertical_flag = 0
    if input_points.shape[1] == 2:
        vertical_flag = 1
        input_points = np.transpose(input_points)
    B = input_points.shape[1]
    input_points = np.concatenate([
        input_points[1:2, :], 
        input_points[0:1, :],
        np.zeros([1, B]), np.ones([1, B])], axis=0)
    input_points[0, :] = input_points[0, :] * 2.5 + x_offset - 300
    input_points[1, :] = input_points[1, :] * 2.5 + y_offset - 900
    input_points[2, :] += z_offset
    
    # print(intrinsic_mat.shape, extrinsic_mat.shape, input_points.shape)
    
    output_points = intrinsic_mat @ extrinsic_mat @ input_points
    output_points = output_points[:2, :] / output_points[2, :]
    if vertical_flag:
        output_points = np.transpose(output_points)
    return output_points

def get_2d_to_3d_mat(intrinsic_mat, extrinsic_mat):
    threeD2twoD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    greed_T_3d = np.array([[0, 2.5, -300], [2.5, 0, -900], [0, 0, 1]])
    greed_T_3d = np.linalg.inv(greed_T_3d)
    
    return greed_T_3d @ np.linalg.inv(project_mat)
    

def compute_size(target_size, image_size):
    '''
    compute resized size of the image in the target size
    preserving ratio of the original image
    returns resized size and offset
    '''
    target_w, target_h = target_size
    image_w, image_h = image_size
    w_ratio = target_w / image_w
    h_ratio = target_h / image_h
    if w_ratio < h_ratio:
        resized_w = target_w
        resized_h = int(image_h * w_ratio)
        offset = (0, (target_h - resized_h) // 2)
    else:
        resized_w = int(image_w * h_ratio)
        resized_h = target_h
        offset = ((target_w - resized_w) // 2, 0)
    return (resized_w, resized_h), offset

    
def plot(path):
    data = np.genfromtxt(path, delimiter=",")
    # data = data[:, (0, 1, 7 ,8)]
    data = data[:, (1, 2, 8, 9)]
    
    # data[:, [2, 3]] = data[:, [3, 2]]

    ids = np.unique(data[:, 1]).astype(int).tolist()
    frames = np.unique(data[:, 0]).astype(int).tolist()
    
    # colors = {id: mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[15 * idx]] for idx, id in enumerate(ids)}
    colors = {id: mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx]] for idx, id in enumerate(ids)}
    
    camera_ids = {0: 'C1', 1: 'C2', 2: 'C3'}
    # seq = '01'
    seq = '02'
    
    image_paths = {
        camera_id: sorted(
            [ osp.join(DATA_PATH, f'images/cam{camera_id+1}/{seq}/cam{camera_id+1}_{seq}_{int(frame)+1}.jpg')
                for frame in frames ]
            ) for camera_id in camera_ids.keys() }
    
    for cam_id in camera_ids.keys():
        intrinsic, extrinsic = get_intrinsic_extrinsic_matrix(cam_id)
        center_points = project_2d_points(intrinsic, extrinsic, data[:, 2:4], 0, 0, 190)
        box_points = [
            project_2d_points(intrinsic, extrinsic, data[:, 2:4], x_off, y_off, z_off)
            for z_off in [0, 177]
            for y_off in [-50, 50]
            for x_off in [-50, 50]
        ]
        data = np.concatenate([data, center_points, *box_points], axis=1)
        
        
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(13, 6), dpi=100)
    fig.set_facecolor('black')
    plt.axis('off')
    
    plt.xlim(0, 1440)
    plt.ylim(0, 480)
    
    for frame_idx, frame in enumerate(frames):
        
        mosaic = Image.new('RGB', size=(1920, 1080), color=(128, 128, 128))
        frame_save_path = osp.join(SAVE_PATH, 'mosaic', f'{frame_idx:04d}.png')
    
        
        frame_data = data[data[:, 0] == frame]
        
        for id in ids:
            id_data = data[data[:, 1] == id]
            id_data = id_data[id_data[:, 0] <= frame]
            
            plt.plot(id_data[:, 2], id_data[:, 3], linewidth=5, color=colors[id])
            plt.plot(id_data[-1:, 2], id_data[-1:, 3], marker='o', markersize=10, color=colors[id])
            
            if id_data.shape[0] == 0:
                continue
            
            plt.text(id_data[-1, 2], id_data[-1, 3], f' id={id}', color=colors[id], fontsize='xx-large')
        
        w_file_path = osp.join(SAVE_PATH, f'plot_pred_{frame}.png')
        
        plt.savefig(w_file_path, dpi=100, bbox_inches='tight')
        plt.cla()
        
        w_image = Image.open(w_file_path)
        
        w_size, w_offset = compute_size((1920//2, 1080//2), w_image.size)
        w_image = w_image.resize(w_size)
        mosaic.paste(w_image, (w_offset[0]+1920//2, w_offset[1]+1080//2))
        
        for cam_id in camera_ids.keys():
            
            image = Image.open(image_paths[cam_id][frame_idx])
            
            # for id in ids:
                
            #     cam_index = 4 + cam_id*18
            #     draw = ImageDraw.Draw(image)
                
            #     line_data = data[data[:, 1] == id]
            #     line_data = line_data[line_data[:, 0] <= frame]
            #     line_data = line_data[line_data[:, cam_index] >= 0]
            #     line_data = line_data[line_data[:, cam_index] < 1920]
            #     line_data = line_data[line_data[:, cam_index+1] >= 0]
            #     line_data = line_data[line_data[:, cam_index+1] < 1080]
                
            #     if line_data.shape[0] != 0:
            #         draw.line(line_data[:, cam_index:cam_index+2].flatten().tolist(), fill=colors[id], width=10)
            
            for id in ids:
                
                cam_index = 4 + cam_id*18
                draw = ImageDraw.Draw(image)
                
                id_data = frame_data[frame_data[:, 1] == id].flatten() # [1, 22]
                
                if id_data.shape[0] == 0:
                    continue
        
                
                if (id_data[cam_index]<0).any() or (id_data[cam_index]>=1920).any() or \
                    (id_data[cam_index+1]<0).any() or (id_data[cam_index+1]>=1080).any():
                    continue
                
                draw.ellipse([id_data[cam_index]-20, id_data[cam_index+1]-20, id_data[cam_index]+20, id_data[cam_index+1]+20], fill=colors[id])
                draw.text([id_data[cam_index]+20, id_data[cam_index+1]-30], f'id_{id}', colors[id], font=font, stroke_width=1)
                
                for i, j in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    x_i = id_data[cam_index+i*2+2]
                    y_i = id_data[cam_index+i*2+3]
                    x_j = id_data[cam_index+j*2+2]
                    y_j = id_data[cam_index+j*2+3]
                    draw.line([x_i, y_i, x_j, y_j], fill=colors[id], width=5)
                
            
            image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}.png'))
            mosaic.paste(image.resize((1920//2, 1080//2)), (1920//2*(cam_id%2), 1080//2*(cam_id//2)))
            
            intrinsic, extrinsic = get_intrinsic_extrinsic_matrix(cam_id)
            warp_proj = get_2d_to_3d_mat(intrinsic, extrinsic)
            warped_image = warp_perspective(
                torch.tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0,
                torch.tensor(warp_proj).unsqueeze(0).float(), 
                (480, 1440))
            
            warped_image = warped_image.squeeze().permute(1, 2, 0).numpy() * 255
            warped_image = Image.fromarray(warped_image.astype(np.uint8))
            warped_image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}_warped.png'))
            
            # mosaic.paste(warped_image.resize((1920//3, 1080//3)), (1920//3*(cam_id%3), 1080//3*(cam_id//3)+1080//3))
        
        mosaic.save(frame_save_path)
    
    os.system(f'ffmpeg -framerate 2 -pattern_type glob -i "{SAVE_PATH}/mosaic/*.png"  {SAVE_PATH}/mosaic/result.mp4')
            
            
            # exit()
                
        

if __name__ == '__main__':
    # path = '../../data/cache/mota_gt.txt'
    # path = '../../data/cache/mota_pred.txt'
    # path = '../lightning_logs/version_5/mota_pred.txt'
    path = '../lightning_logs/version_37/mota_pred.txt'
    plot(path)
