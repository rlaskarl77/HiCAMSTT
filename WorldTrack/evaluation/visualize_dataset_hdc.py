import json
import os
import os.path as osp
import cv2
import numpy as np
import argparse

from glob import glob
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from kornia.geometry.transform.imgwarp import warp_perspective

import torch


intrinsic_camera_matrix_filenames = ['intr_cam1.xml', 'intr_cam2.xml', 'intr_cam3.xml']
extrinsic_camera_matrix_filenames = ['extr_cam1.xml', 'extr_cam2.xml', 'extr_cam3.xml']

DATA_PATH = '/131_data/datasets/HiCAMS/20240415'
SAVE_PATH = '/home/namgi/TrackTacular/visualization/hdc/dataset_rotated'
PRED_FILE = 'mota_pred.txt'

CAM1_ROTATED = True

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
        input_points[0:1, :],
        input_points[1:2, :], 
        np.zeros([1, B]), np.ones([1, B])], axis=0)
    input_points[0, :] = input_points[0, :] + x_offset
    input_points[1, :] = input_points[1, :] + y_offset
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
    greed_T_3d = np.array([[1, 0, 450], [0, 1, 450], [0, 0, 1]])
    
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

def read_ann(path):
    
    cams = { 0: 'cam1', 1: 'cam2', 2: 'cam3' }
    
    frame_id = int(osp.basename(path).split('.')[0])
    ann_file = open(path, 'r')
    anns = json.load(ann_file)
    # for ann in anns:
    #     ann['frame'] = frame_id
    #     ann['id'] = int(ann['personID'][6:])
    #     ann['world_pos'] = ann['positionID']
    #     ann['world_pos'] = np.array([float(ann['world_pos'][:4]), float(ann['world_pos'][4:8])])
    #     ann['world_pos'] = np.array([ann['world_pos'][0] - 1000, ann['world_pos'][1] - 1000])
    #     ann['world_pos'] = ann['world_pos']
    #     ann['img_pos'] = {}
    #     for cam_id in cams.keys():
    #         ann['img_pos'][cams[cam_id]] = np.array([-1, -1, -1, -1])
    #         for view in ann['views']:
    #             if view['viewNum'] == cam_id+1:
    #                 ann['img_pos'][cams[cam_id]] = np.array([
    #                     view['xmin'],
    #                     view['ymin'],
    #                     view['xmax'],
    #                     view['ymax']
    #                 ])
    for ann in anns:
        ann['frame'] = frame_id
        ann['id'] = ann['personID']
        ann['world_pos'] = ann['positionID']
        ann['world_pos'] = np.array([ann['positionID']//900, ann['positionID']%900])
        ann['world_pos'] = np.array([ann['world_pos'][0] - 450, ann['world_pos'][1] - 450])
        ann['world_pos'] = ann['world_pos']
        ann['img_pos'] = {}
        for cam_id in cams.keys():
            ann['img_pos'][cams[cam_id]] = np.array([-1, -1, -1, -1])
            for view in ann['views']:
                if view['viewNum'] == cam_id:
                    ann['img_pos'][cams[cam_id]] = np.array([
                        view['xmin'],
                        view['ymin'],
                        view['xmax'],
                        view['ymax']
                    ])
    
    
    return anns

def parse_all_anns_to_data(all_anns):
    data = []
    for anns in all_anns:
        for ann in anns:
            data.append([
                ann['frame'],
                ann['id'],
                ann['world_pos'][0],
                ann['world_pos'][1],
                ann['img_pos']['cam1'][0],
                ann['img_pos']['cam1'][1],
                ann['img_pos']['cam1'][2],
                ann['img_pos']['cam1'][3],
                ann['img_pos']['cam2'][0],
                ann['img_pos']['cam2'][1],
                ann['img_pos']['cam2'][2],
                ann['img_pos']['cam2'][3],
                ann['img_pos']['cam3'][0],
                ann['img_pos']['cam3'][1],
                ann['img_pos']['cam3'][2],
                ann['img_pos']['cam3'][3]
            ])
    
    return np.array(data)
    
def plot():
    
    ann_paths = sorted(glob(osp.join(DATA_PATH, 'annotations_positions', '*.json')))
    all_anns = [read_ann(ann_path) for ann_path in ann_paths]
    
    data = parse_all_anns_to_data(all_anns)
    
    frames = np.unique(data[:, 0]).astype(int).tolist()
    ids = np.unique(data[:, 1]).astype(int).tolist()
    
    colors = {id: mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx*3]] for idx, id in enumerate(ids)}
    
    camera_ids = {0: 'cam1', 1: 'cam2', 2: 'cam3'}
    
    image_paths = {
        camera_id: sorted(
            [ osp.join(DATA_PATH, f'images/{camera_ids[camera_id]}/{int(frame):08d}.jpg')
                for frame in frames ]
            ) for camera_id in camera_ids.keys() }
    
    for cam_id in camera_ids.keys():
        intrinsic, extrinsic = get_intrinsic_extrinsic_matrix(cam_id)
        center_points = project_2d_points(intrinsic, extrinsic, data[:, 2:4], 0, 0, 0)
        box_points = [
            project_2d_points(intrinsic, extrinsic, data[:, 2:4], x_off, y_off, z_off)
            for z_off in [0, 177]
            for y_off in [-50, 50]
            for x_off in [-50, 50]
        ]
        data = np.concatenate([data, center_points, *box_points], axis=1)
        
        
    # plt.rcParams['axes.facecolor'] = 'black'
    
    # plt.axis('off')
        
    xlim_min, xlim_max = -600, 600
    ylim_min, ylim_max = -600, 600
    
    
    for frame_idx, frame in tqdm(enumerate(frames), total=len(frames)):
        
        mosaic_size = (1920, 1080)
        subplot_size = (1920//3, 1080//3)
        subplots_per_row = 3
        
        mosaic = Image.new('RGB', size=(1920, 1080), color=(128, 128, 128))
        
        if not osp.exists(osp.join(SAVE_PATH, 'mosaic')):
            os.makedirs(osp.join(SAVE_PATH, 'mosaic'))
        
        frame_save_path = osp.join(SAVE_PATH, 'mosaic', f'{frame_idx:04d}.png')
        
        
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        
        ax.set_xlim([xlim_min, xlim_max])
        ax.set_ylim([ylim_min, ylim_max])
        
        ax.invert_yaxis()
        
        
        
        ax.plot([0, 0], [-450, 450], color='white', linewidth=2)
        ax.plot([-450, 450], [0, 0], color='white', linewidth=2)
        
        ax.text(-450, 450, f'frame={frame}', color='white', fontsize='xx-large')
        
        ax.text(0, 0, f'(0m, 0m)', color='white', fontsize='xx-large')
        ax.text(-450, 0, f'(-4.5m, 0m)', color='white', fontsize='xx-large')
        ax.text(450, 0, f'(4.5m, 0m)', color='white', fontsize='xx-large')
        ax.text(0, 450, f'(0m, 4.5m)', color='white', fontsize='xx-large')
        ax.text(0, -450, f'(0m, -4.5m)', color='white', fontsize='xx-large')
    
        for id in ids:
            id_data = data[data[:, 1] == id]
            id_data = id_data[id_data[:, 0] <= frame]
            id_data = id_data[id_data[:, 2] < xlim_max]
            id_data = id_data[id_data[:, 2] > xlim_min]
            id_data = id_data[id_data[:, 3] < ylim_max]
            id_data = id_data[id_data[:, 3] > ylim_min]
            
            ax.plot(id_data[:, 2], id_data[:, 3], linewidth=5, color=colors[id])
            ax.plot(id_data[-1:, 2], id_data[-1:, 3], marker='o', markersize=10, color=colors[id])
            
            if id_data.shape[0] == 0:
                continue
            
            ax.text(id_data[-1, 2], id_data[-1, 3], f' id={id}', color=colors[id], fontsize='xx-large')
        
        
        
        # draw x, y axis and origin
        
        ax.plot([-450, 450], [-450, -450], color='green', linewidth=4)
        ax.plot([450, 450], [450, -450], color='green', linewidth=4)
        ax.plot([-450, -450], [450, -450], color='green', linewidth=4)
        ax.plot([-450, 450], [450, 450], color='green', linewidth=4)
        
        ax.plot(0, 0, marker='o', markersize=10, color='white')
        ax.set_axis_off()
        
        
        w_file_path = osp.join(SAVE_PATH, f'plot_pred_{frame}.png')
        
        fig.savefig(w_file_path, dpi=100, bbox_inches='tight')
        
        ax.clear()
        fig.clear()
        plt.close(fig)
        
        w_image = Image.open(w_file_path)
        
        w_size, w_offset = compute_size(subplot_size, w_image.size)
        w_image = w_image.resize(w_size)
        mosaic.paste(w_image, (w_offset[0]+subplot_size[0]*1, w_offset[1]+subplot_size[1]*2))
        
        
        frame_data = data[data[:, 0] == frame]
        
        for cam_id in camera_ids.keys():
            
            image = Image.open(image_paths[cam_id][frame_idx])
            draw = ImageDraw.Draw(image)
            
            for id in ids:
                
                cam_index = 16 + cam_id*18
                
                id_data = frame_data[frame_data[:, 1] == id].flatten()
                
                if id_data.shape[0] == 0:
                    continue
                
                bbox = id_data[4+cam_id*4:8+cam_id*4]
                bbox = [int(x) for x in bbox]
                
                if (bbox[0]<0 or bbox[0]>=1920) or (bbox[1]<0 or bbox[1]>=1080) or \
                    (bbox[2]<0 or bbox[2]>=1920) or (bbox[3]<0 or bbox[3]>=1080):
                    continue
                
                draw.rectangle(bbox, outline=colors[id], width=5)
                
                foot = [(bbox[0]+bbox[2])//2, bbox[3]] if (CAM1_ROTATED or cam_id != 0) else [bbox[2], (bbox[1]+bbox[3])//2]
                
                draw.ellipse([foot[0]-20, foot[1]-20, foot[0]+20, foot[1]+20], fill=colors[id])
                draw.text([foot[0]+20, foot[1]-30], f'id_{id}', colors[id], font=font, stroke_width=1)
        
                
                if (id_data[cam_index]<0).any() or (id_data[cam_index]>=1920).any() or \
                    (id_data[cam_index+1]<0).any() or (id_data[cam_index+1]>=1080).any():
                    continue
                
                # draw.ellipse([id_data[cam_index]-20, id_data[cam_index+1]-20, id_data[cam_index]+20, id_data[cam_index+1]+20], fill=colors[id])
                # draw.text([id_data[cam_index]+20, id_data[cam_index+1]-30], f'id_{id}', colors[id], font=font, stroke_width=1)
                
                # for i, j in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                #     x_i = id_data[cam_index+i*2+2]
                #     y_i = id_data[cam_index+i*2+3]
                #     x_j = id_data[cam_index+j*2+2]
                #     y_j = id_data[cam_index+j*2+3]
                #     draw.line([x_i, y_i, x_j, y_j], fill=colors[id], width=5)
            
            image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}.png'))
            
            mosaic.paste(image.resize(subplot_size), 
                         (subplot_size[0]*(cam_id%subplots_per_row), 
                          subplot_size[1]*(cam_id//subplots_per_row)))
            
            
            image = Image.open(image_paths[cam_id][frame_idx])
            
            intrinsic, extrinsic = get_intrinsic_extrinsic_matrix(cam_id)
            warp_proj = get_2d_to_3d_mat(intrinsic, extrinsic)
            warped_image = warp_perspective(
                torch.tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0,
                torch.tensor(warp_proj).unsqueeze(0).float(), 
                (900, 900))
            
            warped_image = warped_image.squeeze().permute(1, 2, 0).numpy() * 255
            warped_image = Image.fromarray(warped_image.astype(np.uint8))
            
            draw = ImageDraw.Draw(warped_image)
            draw.line([450, 0, 450, 900], fill='white', width=5)
            draw.line([0, 450, 900, 450], fill='white', width=5)
            
            
            for id in ids:
                
                id_data = frame_data[frame_data[:, 1] == id].flatten()
                
                if id_data.shape[0] == 0:
                    continue
                
                foot = id_data[2:4]
                foot = [int(foot[0])+450, int(foot[1])+450]
                
                draw.ellipse([foot[0]-20, foot[1]-20, foot[0]+20, foot[1]+20], fill=colors[id])
                draw.text([foot[0]+20, foot[1]-30], f'id_{id}', colors[id], font=font, stroke_width=1)
            
            # draw boundary
            draw.line([0, 0, 900, 0], fill='green', width=3)
            draw.line([0, 0, 0, 900], fill='green', width=3)
            draw.line([900, 0, 900, 900], fill='green', width=3)
            draw.line([0, 900, 900, 900], fill='green', width=3)
            
                
            warped_image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}_warped.png'))
            
            warped_size, warped_offset = compute_size(subplot_size, warped_image.size)
            mosaic.paste(warped_image.resize(warped_size), 
                         (warped_offset[0]+subplot_size[0]*(cam_id%subplots_per_row), 
                          warped_offset[1]+subplot_size[1]*(cam_id//subplots_per_row+1)))
        
        mosaic.save(frame_save_path)
    
    os.system(f'ffmpeg -framerate 2 -pattern_type glob -i "{SAVE_PATH}/mosaic/*.png"  {SAVE_PATH}/mosaic/result.mp4')
            
            
            # exit()
                

def parse_args():
    parser = argparse.ArgumentParser(description='Plot the tracking result')
    parser.add_argument('--path', type=str, default=PRED_FILE, help='Path to the prediction file')
    return parser.parse_args()
        

if __name__ == '__main__':
    args = parse_args()
    plot()
