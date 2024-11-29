import os
import os.path as osp
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
#                                      'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
# extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
#                                      'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

# DATA_PATH = '/data/namgi/wildtrack/Wildtrack_dataset'
# IMAGE_PATH = '/data/namgi/wildtrack/Wildtrack_dataset/Image_subsets/C1/00000000.png'
# SAVE_PATH = '/home/namgi/TrackTacular/visualization'
# PRED_FILE = '/home/namgi/TrackTacular/WorldTrack/lightning_logs/version_5/mota_pred.txt'

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']

DATA_PATH = '/data/namgi/HiCAMS/20241022/07/'
SAVE_PATH = '/home/namgi/HiCAMSTT/visualization/factory/07'
PRED_FILE = '/home/namgi/HiCAMSTT/exp/lightning_logs/test/factory/mvdet/baseline/mota_pred.txt'

font_size = 50
font = ImageFont.truetype("Arial.ttf", font_size)
font_color = (255, 64, 0)


def get_worldgrid_from_pos(pos):
    grid_x = pos // 10000
    grid_y = pos % 10000
    return np.array([grid_x, grid_y], dtype=int)

def get_intrinsic_extrinsic_matrix(camera_i):
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
    input_points[0, :] = input_points[0, :] * 0.1 - 5
    input_points[1, :] = input_points[1, :] * 0.1 + 90
    input_points[0, :] = input_points[0, :] + x_offset
    input_points[1, :] = input_points[1, :] + y_offset
    input_points[2, :] += z_offset
    
    output_points = intrinsic_mat @ extrinsic_mat @ input_points
    output_points = output_points[:2, :] / output_points[2, :]
    if vertical_flag:
        output_points = np.transpose(output_points)
    return output_points

def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=1):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    threeD2twoD = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, z], [0, 0, 1]])
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat

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

    data = data[:, (1, 2, 8, 9)]

    ids = np.unique(data[:, 1]).astype(int).tolist()
    frames = np.unique(data[:, 0]).astype(int).tolist()
    
    colors = {id: mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx]] for idx, id in enumerate(ids)}
    
    camera_ids = {0: 'cam1', 1: 'cam2'}
    # image_paths = {camera_id: sorted([osp.join(DATA_PATH, f'Image_subsets/{camera_ids[camera_id]}/{int(frame*5):08d}.png') for frame in frames]) for camera_id in camera_ids.keys()}
    image_paths = {camera_id: sorted([osp.join(DATA_PATH, f'Image_subsets/{camera_ids[camera_id]}/{int(frame):06d}.jpg') for frame in frames]) for camera_id in camera_ids.keys()}
    
    for cam_id in camera_ids.keys():
        intrinsic, extrinsic = get_intrinsic_extrinsic_matrix(cam_id)
        center_points = project_2d_points(intrinsic, extrinsic, data[:, 2:4], 0, 0, 1.90)
        box_points = [
            project_2d_points(intrinsic, extrinsic, data[:, 2:4], x_off, y_off, z_off)
            for z_off in [0, 1.77]
            for y_off in [-.50, .50]
            for x_off in [-.50, .50]
        ]
        data = np.concatenate([data, center_points, *box_points], axis=1)
        
        
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(13, 6), dpi=100)
    fig.set_facecolor('black')
    plt.axis('off')
    
    for frame_idx, frame in enumerate(frames):
        
        mosaic = Image.new('RGB', size=(1920, 1080), color=(128, 128, 128))
        frame_save_path = osp.join(SAVE_PATH, f'vid/{frame_idx:04d}.png')
    
        
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
        
        w_size, w_offset = compute_size((1920//2*2, 1080//2), w_image.size)
        w_image = w_image.resize(w_size)
        mosaic.paste(w_image, (w_offset[0], w_offset[1]+1080//2))
        
        for cam_id in camera_ids.keys():
            
            image = Image.open(image_paths[cam_id][frame_idx])
            
            for id in ids:
                
                cam_index = 4 + cam_id*18
                draw = ImageDraw.Draw(image)
                
                line_data = data[data[:, 1] == id]
                line_data = line_data[line_data[:, 0] <= frame]
                line_data = line_data[line_data[:, cam_index] >= 0]
                line_data = line_data[line_data[:, cam_index] < 1920]
                line_data = line_data[line_data[:, cam_index+1] >= 0]
                line_data = line_data[line_data[:, cam_index+1] < 1080]
                
                if line_data.shape[0] != 0:
                    draw.line(line_data[:, cam_index:cam_index+2].flatten().tolist(), fill=colors[id], width=10)
            
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
                
            
            # image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}.png'))
            mosaic.paste(image.resize((1920//2, 1080//2)), (1920//2*(cam_id%2), 1080//2*(cam_id//2)))
        
        mosaic.save(frame_save_path)
    
    os.system(f'ffmpeg -framerate 2 -pattern_type glob -i "{SAVE_PATH}/vid/*.png"  {SAVE_PATH}/vid/result.mp4')
            
            
            # exit()
                
        

if __name__ == '__main__':
    # path = '../../data/cache/mota_gt.txt'
    # path = '../../data/cache/mota_pred.txt'
    # path = '../lightning_logs/version_5/mota_pred.txt'
    path = '../lightning_logs/version_10/mota_pred.txt'
    plot(PRED_FILE)
