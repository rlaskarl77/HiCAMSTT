from io import BytesIO
import os
import os.path as osp
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')  # GUI 없이 동작하도록 설정
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import concurrent.futures
import random
from tqdm import tqdm

# DATA_PATH = '/data/namgi/HiCAMS/20241022/03/'
SAVE_PATH = '/home/namgi/HiCAMSTT/visualization/dataset'

font_size = 50
font = ImageFont.truetype("Arial.ttf", font_size)
font_color = (255, 64, 0)
colors = {id: mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[id % 949]] for id in range(1000)}

CAM_PARAMS = {
    "cam63": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.03780599785522,
            2.121679873691477,
            -1.3374365036402904
        ],
        "tvec": [
            -203.1716444195454,
            -74.53229842129745,
            187.7491104563038
        ]
    },
    "cam64": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9927930532158047,
            0.7877987066982112,
            -0.6941865927504685
        ],
        "tvec": [
            -156.92849013275108,
            89.07683058781342,
            -73.6562321696746
        ]
    },
    "cam65": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0980678550373892,
            2.2184484772846442,
            -1.1604302766190677
        ],
        "tvec": [
            -137.48520545941798,
            -65.94336472253306,
            122.10626734402585
        ]
    },
    "cam66": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8159877047382846,
            1.0241604870425753,
            -0.5477138755857991
        ],
        "tvec": [
            -91.18078911145926,
            22.453781104744184,
            -20.087448622042753
        ]
    },
    "cam68": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            2.040945682022718,
            0.8561083357602651,
            -0.5676916301038758
        ],
        "tvec": [
            -14.730008344652514,
            2.9114570957977857,
            26.220477687958013
        ]
    },
    "cam72": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.0711613881232815,
            -2.032238490646261,
            1.178585592158765
        ],
        "tvec": [
            238.09508710896847,
            -76.80438550260332,
            131.30606360817924
        ]
    },
    "cam73": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.9856869409013398,
            -0.7379542542930861,
            0.49885714288722405
        ],
        "tvec": [
            133.71296569471158,
            86.27019253466314,
            -108.57030320299594
        ]
    },
    "cam74": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            0.9661684788441447,
            -2.2732914452355937,
            1.2050721153835684
        ],
        "tvec": [
            137.18635655042075,
            -65.3683602568311,
            121.28470866951317
        ]
    },
    "cam76": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.909945809773252,
            -0.8751251901875915,
            0.5474804445725748
        ],
        "tvec": [
            72.75798482816228,
            40.244222510388106,
            -40.28219527600947
        ]
    },
    "cam78": {
        "mtx": [
            [ 982.761, 0.0, 988.18977 ],
            [ 0.0, 1128.76581, 510.34356 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        "rvec": [
            1.8039614088785945,
            -1.0075587724770085,
            0.5704870659358283
        ],
        "tvec": [
            2.5818068668951097,
            11.432404299131896,
            10.888543976918278
        ]
    }
}

def get_projection_matrix(camera_id):
    cam = f'cam{camera_id:02d}'
    mtx = np.array(CAM_PARAMS[cam]['mtx'])
    rvec = np.array(CAM_PARAMS[cam]['rvec'])
    tvec = np.array(CAM_PARAMS[cam]['tvec'])
    rmat = cv2.Rodrigues(rvec)[0]
    ext = np.concatenate([rmat, tvec.reshape(-1, 1)], axis=1) # 3x4
    # drop z-axis
    ext = ext[:, [0, 1, 3]]
    
    P = mtx @ ext
    P_inv = np.linalg.pinv(P)
    
    return P, P_inv

def clip_line(p1, p2, image_width, image_height):
    # Cohen-Sutherland line clipping algorithm
    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    def compute_code(x, y):
        code = INSIDE
        if x < 0:
            code |= LEFT
        elif x >= image_width:
            code |= RIGHT
        if y < 0:
            code |= BOTTOM
        elif y >= image_height:
            code |= TOP
        return code

    def clip_point():
        nonlocal x1, y1, x2, y2, code1, code2
        
        # If both points are outside in same region, reject
        if (code1 & code2) != 0:
            return False

        # If both points are inside, accept
        if code1 == 0 and code2 == 0:
            return True

        # Pick outside point
        code_out = code2 if code1 == 0 else code1

        # Find intersection point
        if code_out & TOP:
            x = x1 + (x2 - x1) * (image_height - 1 - y1) / (y2 - y1)
            y = image_height - 1
        elif code_out & BOTTOM:
            x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            y = 0
        elif code_out & RIGHT:
            y = y1 + (y2 - y1) * (image_width - 1 - x1) / (x2 - x1)
            x = image_width - 1
        elif code_out & LEFT:
            y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
            x = 0

        # Replace outside point with intersection point
        if code_out == code1:
            x1, y1 = x, y
            code1 = compute_code(x1, y1)
        else:
            x2, y2 = x, y
            code2 = compute_code(x2, y2)

        return True

    x1, y1 = p1
    x2, y2 = p2
    accepted = False
    
    while True:
        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)
        
        if (code1 | code2) == 0:  # Both points inside
            accepted = True
            break
        elif (code1 & code2) != 0:  # Both points outside
            break
        else:
            if not clip_point():  # Failed to find intersection
                break
                
    if accepted:
        return [(int(x1), int(y1)), (int(x2), int(y2))]
    return None

def draw_grid_on_ground(draw, camera_id, scene_range, color='white', grid_size=5):
    P, _ = get_projection_matrix(camera_id)
    
    image_width = 1920
    image_height = 1080
    
    # Use scene range for grid boundaries
    scene_minX, scene_maxX = scene_range[0], scene_range[1]
    scene_minY, scene_maxY = scene_range[2], scene_range[3]
    
    x_range = np.arange(scene_minX, scene_maxX + grid_size, grid_size)
    y_range = np.arange(scene_minY, scene_maxY + grid_size, grid_size)
    
    # Project all points first
    grid_points = {}
    for x in x_range:
        for y in y_range:
            point_3d = np.array([x, y, 1])
            point_2d = P @ point_3d
            point_2d = point_2d / point_2d[2]
            grid_points[(x, y)] = (int(point_2d[0]), int(point_2d[1]))
    
    # Draw vertical lines
    for x in x_range:
        for y1, y2 in zip(y_range[:-1], y_range[1:]):
            p1 = grid_points[(x, y1)]
            p2 = grid_points[(x, y2)]
            clipped = clip_line(p1, p2, image_width, image_height)
            if clipped:
                draw.line(clipped, fill=color, width=2)
        
        # Add distance label at the bottom and top edges
        if x % 10 == 0:  # Label every 10m
            p = grid_points[(x, scene_minY)]
            if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
                draw.text((p[0]-10, p[1]-20), f'{(int(x), int(scene_minY))}', fill=color, font=font)
            p = grid_points[(x, scene_maxY)]
            if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
                draw.text((p[0]-10, p[1]-20), f'{(int(x), int(scene_maxY))}', fill=color, font=font)
    
    # Draw horizontal lines
    for y in y_range:
        for x1, x2 in zip(x_range[:-1], x_range[1:]):
            p1 = grid_points[(x1, y)]
            p2 = grid_points[(x2, y)]
            clipped = clip_line(p1, p2, image_width, image_height)
            if clipped:
                draw.line(clipped, fill=color, width=2)
        
        # Add distance label at the left and right edges
        if y % 10 == 0:  # Label every 10m
            p = grid_points[(scene_minX, y)]
            if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
                draw.text((p[0]+5, p[1]-10), f'{(int(scene_minX), int(y))}', fill=color, font=font)
            p = grid_points[(scene_maxX, y)]
            if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
                draw.text((p[0]+5, p[1]-10), f'{(int(scene_maxX), int(y))}', fill=color, font=font)

def select_path_and_params(scene):
    data = {}
    if scene == '01':
        data['frame_range'] = range(1, 341) # 1 ~ 340
        data['cameras'] = {64: 0, 73: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 190, 240]
    elif scene == '02':
        data['frame_range'] = range(1, 260) # 1 ~ 261
        data['cameras'] = {65: 0, 74: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 150, 200]
    elif scene == '03':
        data['frame_range'] = range(30, 118) # 3 ~ 117
        data['cameras'] = {65: 0, 74: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 150, 200]
    elif scene == '05':
        data['frame_range'] = range(1, 251) # 1 ~ 250
        data['cameras'] = {68: 0, 78: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 0, 50]
    elif scene == '06':
        data['frame_range'] = range(1, 247) # 1 ~ 246
        data['cameras'] = {64: 0, 73: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 190, 240]
    elif scene == '07':
        data['frame_range'] = range(2, 367) # 2 ~ 366
        data['cameras'] = {65: 0, 74: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100. - 200. + 5) * 10) * 10000 + int((y/100. - 200. - 90) * 10)
        data['scene_range'] = [0, 25, 150, 200]
    elif scene == '08':
        data['frame_range'] = range(1, 172) # 1 ~ 171
        data['cameras'] = {68: 0, 78: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 0, 50]
    elif scene == '09':
        data['frame_range'] = range(1, 93) # 1 ~ 92
        data['cameras'] = {68: 0, 78: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 0, 50]
    elif scene == '10':
        data['frame_range'] = range(1, 130) # 1 ~ 129
        data['cameras'] = {66: 0, 76: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 85, 135]
    elif scene == '11':
        data['frame_range'] = range(1, 259) # 1 ~ 258
        data['cameras'] = {66: 0, 76: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 85, 135]
    elif scene == '12':
        data['frame_range'] = range(1, 86) # 1 ~ 85
        data['cameras'] = {65: 0, 74: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 150, 200]
    elif scene == '13':
        data['frame_range'] = range(1, 166) # 1 ~ 165
        data['cameras'] = {65: 0, 74: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 150, 200]
    elif scene == '14':
        data['frame_range'] = range(1, 261) # 1 ~ 260
        data['cameras'] = {63: 0, 72: 1}
        data['parse_position'] = lambda x, y: \
            int((x/100.) * 10) * 10000 + int((y/100.) * 10)
        data['scene_range'] = [0, 25, 240, 290]
        
    data['mask_posID'] = lambda x, y: \
        (x >= data['scene_range'][0]) and (x <= data['scene_range'][1]) \
            and (y >= data['scene_range'][2]) and (y <= data['scene_range'][3])

    cams = list(data['cameras'].keys())
    cam1 = f'cam{cams[0]}'
    cam2 = f'cam{cams[1]}'
    data['scene_dir'] = f'/data/namgi/HiCAMS/20241022/{scene}'
    data['json_prefix'] = os.path.join(data['scene_dir'], f'tracked/{scene}_')
    data['image_prefix1'] = os.path.join(data['scene_dir'], f'images2/{cam1}/{cam1}_{scene}_')
    data['image_prefix2'] = os.path.join(data['scene_dir'], f'images2/{cam2}/{cam2}_{scene}_')
    data['image_dir'] = os.path.join(data['scene_dir'], f'Image_subsets')
    data['json_dir'] = os.path.join(data['scene_dir'], f'annotations_positions')
    
    return data

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

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
    
def plot(data):
    
    print('Plotting...')
    
    annotation_dir = data['json_dir']
    json_paths = [osp.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.json')]
    json_paths = sorted(json_paths)
    
    frames = [int(f.split('/')[-1].split('.')[0]) for f in json_paths]
    
    jsons = {frame: read_json(json_path) for frame, json_path in zip(frames, json_paths)}
    
    cameras = data['cameras']
    image_paths = {
        camera_id: sorted([
            osp.join(data['image_dir'], f'cam{cameras[camera_id]+1}', f'{int(frame):06d}.jpg') 
            for frame in frames
        ]) for camera_id in cameras.keys()
    }
    
    
    scene = data['scene_dir'].split('/')[-1]
    save_dir = osp.join(SAVE_PATH, scene)
    
    scene_range = data['scene_range']
    
    scene_minX, scene_maxX = scene_range[0], scene_range[1]
    scene_minY, scene_maxY = scene_range[2], scene_range[3]
    
    # print(f'Processing {scene}')
    
    for frame_idx, frame in tqdm(enumerate(frames), total=len(frames)):
        fig = plt.figure(figsize=(13, 6), dpi=100)
        fig.set_facecolor('black')
        ax = plt.gca()
        ax.set_facecolor('black')

        # Subplot 여백 수동 설정
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        
        # 축 비율을 1:1로 설정
        ax.set_aspect('equal')
        
        # x, y 축 간격을 5로 설정
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        
        # Set axis limits based on scene_range
        ax.set_ylim(data['scene_range'][0], data['scene_range'][1])
        ax.set_xlim(data['scene_range'][2], data['scene_range'][3])

        mosaic = Image.new('RGB', size=(1920, 1080), color=(128, 128, 128))
        
        if not os.path.exists(osp.join(save_dir, 'vid')):
            os.makedirs(osp.join(save_dir, 'vid'))
        frame_save_path = osp.join(save_dir, f'vid/{frame_idx:04d}.png')
    
        
        frame_data = jsons[frame]
        
        for ann in frame_data:
            id = ann['personID']
            positionID = ann['positionID']
            posX, posY = positionID // 10000, positionID % 10000
            if scene == '07':
                posX, posY = posX * 0.1 - 5, posY * 0.1 + 90
            else:
                posX, posY = posX * 0.1, posY * 0.1
                
            posX, posY = posX - 200, posY - 200
            
            if not data['mask_posID'](posX, posY):
                continue
            
            plt.plot(posY, posX, marker='o', markersize=10, color=colors[id])
            plt.text(posY, posX, f' id={id}', color=colors[id], fontsize='xx-large')
        
        # 그리드 설정을 더 명확하게
        plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # x, y 축 레이블 설정
        plt.xlabel('Y Position', color='white', fontsize=12)
        plt.ylabel('X Position', color='white', fontsize=12)
        
        # 축 스타일 설정
        plt.tick_params(axis='x', colors='white', labelsize=10)
        plt.tick_params(axis='y', colors='white', labelsize=10)
        
        # y축 반전
        plt.gca().invert_yaxis()
        
        # 여백 조정
        plt.tight_layout()
        
        # Convert plot to image and paste on mosaic
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_image = Image.open(buf)
        w_size, w_offset = compute_size((1920//2*2, 1080//2), plot_image.size)
        plot_image = plot_image.resize(w_size)
        mosaic.paste(plot_image, (w_offset[0], w_offset[1]+1080//2))
        
        plt.close(fig)
        
        # print(f'Frame {frame_idx} done')
        
        for cam_id in cameras.keys():
            
            # print(f'Processing {frame_idx} {cam_id}')
            
            image = Image.open(image_paths[cam_id][frame_idx])
            draw = ImageDraw.Draw(image)
            
            draw_grid_on_ground(draw, cam_id, scene_range)
            
            for ann in frame_data:
                id = ann['personID']
                positionID = ann['positionID']
                
                for view in ann['views']:
                    if view['viewNum'] == cameras[cam_id]:
                        
                        xmin = view['xmin']
                        ymin = view['ymin']
                        xmax = view['xmax']
                        ymax = view['ymax']
                        
                        # no label
                        if (xmin, ymin, xmax, ymax) == (-1, -1, -1, -1):
                            continue
                        
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=colors[id], width=5)
                        draw.text((xmin, ymin), f' id={id}', font=font, fill=colors[id])
            
            # image.save(osp.join(SAVE_PATH, f'plot_pred_{frame}_{cam_id}.png'))
            mosaic.paste(image.resize((1920//2, 1080//2)), (1920//2*(cameras[cam_id]%2), 1080//2*(cameras[cam_id]//2)))
        
        mosaic.save(frame_save_path)
    
    os.system(f'ffmpeg -framerate 2 -pattern_type glob -i "{save_dir}/vid/*.png"  {save_dir}/vid/result.mp4')

def process_scene(scene):
    try:
        # 각 프로세스마다 독립적인 난수 시드 설정
        random.seed()
        np.random.seed()
        
        data = select_path_and_params(scene)
        print(f'{scene} start')
        plot(data)
        print(f'{scene} done')
        return f"Scene {scene} completed successfully"
    except Exception as e:
        return f"Error in scene {scene}: {str(e)}"

if __name__ == '__main__':
    scenes = ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
    
    # ThreadPoolExecutor 대신 ProcessPoolExecutor 사용
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Future 객체로 결과 추적
        future_to_scene = {executor.submit(process_scene, scene): scene for scene in scenes}
        
        for future in concurrent.futures.as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f'Scene {scene} generated an exception: {e}')
