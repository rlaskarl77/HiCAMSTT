import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np

def calculate_world_size(xworld_values, yworld_values):
    xworld_min, xworld_max = min(xworld_values), max(xworld_values)
    yworld_min, yworld_max = min(yworld_values), max(yworld_values)
    
    world_width = xworld_max - xworld_min + 1
    world_height = yworld_max - yworld_min + 1
    return world_width, world_height, xworld_min, yworld_min, xworld_max, yworld_max 

def is_in_central_region(x, y, center_x, center_y, width, height):
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

data_dir = "/data/aicity/train"
columns = ['camera_id', 'obj_id', 'frame_id', 'xmin', 'ymin', 'width', 'height', 'xworld', 'yworld']
center_x, center_y = 0, 2.5 
rect_width, rect_height = 10, 15 
max_frame_gap = 10 

cmap = cm.get_cmap('tab10')
colors = {obj_id: cmap(i % 10) for i, obj_id in enumerate(range(1000))}  

for scene_dir in sorted(os.listdir(data_dir)):
    scene_path = os.path.join(data_dir, scene_dir)
    file_path = os.path.join(scene_path, "ground_truth.txt")
    df = pd.read_csv(file_path, sep=' ', names=columns)
    world_width, world_height, xworld_min, yworld_min, xworld_max, yworld_max = calculate_world_size(df['xworld'], df['yworld'])

    # df = df[(df['frame_id'] >= 0) & (df['frame_id'] <= 1000)]

    plt.figure(figsize=(10, 6))

    for obj_id, group in df.groupby('obj_id'):
        group = group.sort_values(by='frame_id')  
        color = colors[obj_id % 10]  

        segment_x, segment_y = [], []
        current_style = None
        last_frame = None  

        for i in range(len(group)):
            x, y = group['xworld'].iloc[i], group['yworld'].iloc[i]
            frame = group['frame_id'].iloc[i]
            in_center = is_in_central_region(x, y, center_x, center_y, rect_width, rect_height)
            new_style = '-' if in_center else ':'

            if current_style is None or new_style == current_style and (last_frame is None or frame - last_frame <= max_frame_gap):
                segment_x.append(x)
                segment_y.append(y)
            else:
                plt.plot(segment_x, segment_y, linestyle=current_style, color=color, marker=None, alpha=0.8, linewidth=1)
                segment_x, segment_y = [x], [y]

            current_style = new_style
            last_frame = frame

        if segment_x and segment_y:
            plt.plot(segment_x, segment_y, linestyle=current_style, color=color, marker=None, alpha=0.8, linewidth=1)

    plt.title('World Grid (xworld, yworld)')
    plt.xlabel('xworld')
    plt.ylabel('yworld')
    plt.xlim(xworld_min, xworld_max)
    plt.ylim(yworld_min, yworld_max)
    plt.grid(True)

    plot_path = f'aicity_grid_vis_center_tracklet/{scene_dir}_world_grid_plot.png'
    plt.savefig(plot_path)
    plt.close()  