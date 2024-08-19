import os
import os.path as osp
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# DATA_PATH = '/data/namgi/MultiviewX/MultiviewX'
# SAVE_PATH = '/home/namgi/TrackTacular/visualization/vid/multiviewx/'
# DATA_PATH = '/data/namgi/wildtrack/Wildtrack_dataset/'
# SAVE_PATH = '/home/namgi/TrackTacular/visualization/vid/wildtrack/'
DATA_PATH = '/131_data/datasets/HiCAMS/20240702/'
SAVE_PATH = '/home/namgi/TrackTacular/visualization/vid/hicams/'

font_size = 50
font = ImageFont.truetype("Arial.ttf", font_size)
font_color = (255, 64, 0)


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

    
def plot():
    
    camera_ids = {id: cam_id for id, cam_id in enumerate(sorted(os.listdir(osp.join(DATA_PATH, 'images'))))}
    frames = sorted([int(f.split('.')[0]) for f in os.listdir(osp.join(DATA_PATH, 'images', camera_ids[0]))])
    num_frames = len(frames)
    frame_ratio = 0.9
    frames = frames[int(num_frames*frame_ratio):]
    
    image_paths = {camera_id: sorted([os.path.join(DATA_PATH, 'images', cam, f)
                                      for f in os.listdir(osp.join(DATA_PATH, 'images', cam))]) 
                   for camera_id, cam in camera_ids.items()}
        
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(13, 6), dpi=100)
    fig.set_facecolor('black')
    plt.axis('off')
    
    if not osp.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    for frame_idx, frame in tqdm(enumerate(frames), total=len(frames)):
        
        mosaic = Image.new('RGB', size=(1920, 1080), color=(128, 128, 128))
        frame_save_path = osp.join(SAVE_PATH, f'{frame_idx:04d}.png')
        
        for cam_id in sorted(camera_ids.keys()):
            
            image = Image.open(image_paths[cam_id][frame_idx])
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f'Camera {cam_id}', font=font, fill=font_color)
            mosaic.paste(image.resize((1920//2, 1080//2)), (1920//2*(cam_id%2), 1080//2*(cam_id//2)))
        
        mosaic.save(frame_save_path)
    
    os.system(f'ffmpeg -framerate 2 -pattern_type glob -i "{SAVE_PATH}/*.png"  {SAVE_PATH}/hicams.mp4')
    
if __name__ == '__main__':
    plot()