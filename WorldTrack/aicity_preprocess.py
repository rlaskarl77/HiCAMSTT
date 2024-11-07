import cv2
import os
import json
import numpy as np
import re

def create_annotation_positions(scene_path, camera_numbers, ground_gt_path, output_dir, grid_shape=(1400, 800), spacing=0.025):
    with open(ground_gt_path, 'r') as f:
        lines = f.readlines()

    num_cameras = len(camera_numbers)
    annotations = {}
    
    xworld_min, xworld_max = -10, 10
    yworld_min, yworld_max = -15, 20

    map_width = 20
    map_height = 35
    map_expand = 40

    x_scale = (xworld_max - xworld_min) / (grid_shape[1] * spacing)
    y_scale = (yworld_max - yworld_min) / (grid_shape[0] * spacing)

    for line in lines:
        camera_id, obj_id, frame_id, xmin, ymin, width, height, xworld, yworld = map(float, line.strip().split(' '))

        frame_id_str = f"{int(frame_id):08d}"

        if frame_id not in annotations:
            annotations[frame_id] = {}

        position_id = get_pos_from_worldcoord((xworld, yworld), xworld_min, yworld_min, map_width, map_expand)
        world_coords = get_worldcoord_from_pos(position_id, xworld_min, yworld_min, map_width, map_expand)

        if obj_id not in annotations[frame_id]:
            annotations[frame_id][obj_id] = {
                "personID": int(obj_id),
                "positionID": int(position_id),
                "views": [{"viewNum": cam_id, "xmax": -1, "xmin": -1, "ymax": -1, "ymin": -1} for cam_id in range(num_cameras)]
            }

        annotations[frame_id][obj_id]["views"][camera_numbers.index(int(camera_id))] = {
            "viewNum": camera_numbers.index(int(camera_id)),
            "xmax": int(xmin + width),
            "xmin": int(xmin),
            "ymax": int(ymin + height),
            "ymin": int(ymin)
        }
         
    for frame_id, persons in annotations.items():
        frame_annotations = list(persons.values())  # Convert to list for the JSON output
        frame_path = os.path.join(output_dir, f"{int(frame_id):08d}.json")
        with open(frame_path, 'w') as f_out:
            json.dump(frame_annotations, f_out, indent=4)

    print(f"Annotation files created for frames in {output_dir}.")

def downsample_ground_truth(ground_gt_path, output_gt_path, original_fps=30, target_fps=2):
    downsample_interval = original_fps // target_fps

    with open(ground_gt_path, 'r') as f:
        lines = f.readlines()

    downsampled_lines = []
    new_frame_id = 1
    first_original_frame_id = None
    last_original_frame_id = None
    lines.sort(key=lambda line: int(float(line.strip().split(' ')[2])))
    
    for line in lines:
        fields = line.strip().split(' ')
        original_frame_id = int(float(fields[2])) 

        if (original_frame_id - 2) % downsample_interval == 0:
            if first_original_frame_id is None:
                first_original_frame_id = original_frame_id  
            last_original_frame_id = original_frame_id      
            
            fields[2] = str(new_frame_id)
            downsampled_lines.append(" ".join(fields) + "\n")
            new_frame_id += 1

    if first_original_frame_id == 2 and last_original_frame_id == 23987:
        print("First and last frame IDs match expected values: 2 and 23987.")
    else:
        print(f"Unexpected first and last frame IDs: {first_original_frame_id} and {last_original_frame_id}")

    with open(output_gt_path, 'w') as f_out:
        f_out.writelines(downsampled_lines)
    print(f"Downsampled Ground Truth saved to {output_gt_path}")

def get_worldgrid_from_worldcoord(world_coord, xworld_min, yworld_min, map_expand=40):
    coord_x, coord_y = world_coord
    grid_x = (coord_x - xworld_min) * map_expand
    grid_y = (coord_y - yworld_min) * map_expand
    return np.array([grid_x, grid_y], dtype=int)

def get_pos_from_worldgrid(worldgrid, map_width, map_expand):
    grid_x, grid_y = worldgrid
    return grid_x + grid_y * map_width * map_expand

def get_pos_from_worldcoord(world_coord, xworld_min, yworld_min, map_width, map_expand):
    grid = get_worldgrid_from_worldcoord(world_coord,xworld_min, yworld_min,map_expand)
    return get_pos_from_worldgrid(grid, map_width, map_expand)

def get_worldgrid_from_pos(pos, map_width, map_expand):
    grid_x = pos % (map_width * map_expand)
    grid_y = pos // (map_width * map_expand)
    return np.array([grid_x, grid_y], dtype=int)

def get_worldcoord_from_worldgrid(worldgrid, xworld_min, yworld_min, map_expand):
    grid_x, grid_y = worldgrid
    coord_x = grid_x / map_expand + xworld_min
    coord_y = grid_y / map_expand + yworld_min
    return np.array([coord_x, coord_y])

def get_worldcoord_from_pos(pos, xworld_min, yworld_min, map_width, map_expand):
    grid = get_worldgrid_from_pos(pos, map_width, map_expand)
    return get_worldcoord_from_worldgrid(grid, xworld_min, yworld_min, map_expand)

def get_camera_numbers(base_path):
    camera_folders = [folder for folder in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, folder)) and "camera" in folder]
    camera_numbers = [int(re.search(r'\d+', folder).group()) for folder in camera_folders]
    return sorted(camera_numbers)


def main():
    data_dir = "/data/aicity/train"
    for scene_dir in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene_dir)
        camera_numbers = get_camera_numbers(scene_path)
        ground_gt_path = os.path.join(scene_path, "ground_truth_fps30.txt")
        downsample_gt_path = os.path.join(scene_path, "ground_truth.txt")
        downsample_ground_truth(ground_gt_path, downsample_gt_path)
        output_dir = os.path.join(scene_path, "annotations_positions")
        os.makedirs(output_dir, exist_ok=True)
        create_annotation_positions(scene_path, camera_numbers, downsample_gt_path, output_dir)

if __name__ == "__main__":
    main()