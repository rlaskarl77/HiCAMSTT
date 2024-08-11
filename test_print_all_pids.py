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


ANNS_PATH = '/131_data/datasets/HiCAMS/20240702/working_files/labeling'
IMGS_PATH = '/131_data/datasets/HiCAMS/20240702/working_files/images'

def reshape_ann(ann_path, save_path):
    cams = { 0: 'cam1', 1: 'cam2', 2: 'cam3' }
    
    with open(ann_path, 'r+', encoding='utf-8') as ann_file:
        anns = json.load(ann_file)
        new_anns = []
        for ann in anns:
            
            new_ann = {}
            new_ann['personID'] = int(ann['personID'][6:])
            new_ann['positionID'] = int(ann['positionID'])
            new_ann['views'] = []
            
            if not is_pedestrain_valid(new_ann['positionID']):
                continue
            
            new_ann['positionID'] = fix_position_id(new_ann['positionID'])
            
            for cam_id in cams.keys():
                view = {}
                view['viewNum'] = cam_id
                view['xmax'] = -1
                view['xmin'] = -1
                view['ymax'] = -1
                view['ymin'] = -1
                new_ann['views'].append(view)
            
            for view in ann['views']:
                cam_id = view['viewNum'] - 1
                new_bbox = [view['xmin'], view['ymin'], view['xmax'], view['ymax']]
                if cam_id == 0:
                    new_bbox = reshape_cam1(*new_bbox)
                new_ann['views'][cam_id]['xmin'] = new_bbox[0]
                new_ann['views'][cam_id]['ymin'] = new_bbox[1]
                new_ann['views'][cam_id]['xmax'] = new_bbox[2]
                new_ann['views'][cam_id]['ymax'] = new_bbox[3]
                
            new_anns.append(new_ann)
    
    if len(new_anns) == 0:
        print(f'No valid pedestrian in {ann_path}')
    with open(save_path, 'w+', encoding='utf-8') as ann_file:
        json.dump(new_anns, ann_file, indent=4)

if __name__ == '__main__':
    
    sequences = sorted(os.listdir(ANNS_PATH))
    
    for seq in sequences:
        ann_paths = set([int(osp.basename(x).split('.')[0].split('_')[-1]) for x in
            glob(osp.join(ANNS_PATH, seq, 'tracked', '*.json'))])
        
        img_paths_all = dict()
        for cam in ['cam1', 'cam2', 'cam3']:
            img_paths = set([int(osp.basename(x).split('.')[0].split('_')[-1]) for x in
                glob(osp.join(IMGS_PATH, cam, seq, '*.jpg'))])
            diff = sl(ann_paths - img_paths)
            if len(diff) > 0:
                print(f'No image for {diff}, cam {seq} {cam}')
            
            diff = sl(img_paths - ann_paths)
            if len(diff) > 0:
                print(f'No annotation for {diff}, {seq} {cam}')
            
            img_paths_all[cam] = img_paths
        
        for cam, cam2 in [('cam1', 'cam2'), ('cam2', 'cam3'), ('cam3', 'cam1')]:
            diff = sl(img_paths_all[cam] - img_paths_all[cam2])
            if len(diff) > 0:
                print(f'No image for {diff}, cam {cam} vs cam {cam2}')
        