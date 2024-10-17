import cv2
import os
from operator import itemgetter


def extract_frames_per_camera(scene_dir, interval_sec=1, fps=30):
    image_subset_dir = os.path.join(scene_dir, "Image_subsets")
    os.makedirs(image_subset_dir, exist_ok=True)
    
    for camera_folder in sorted(os.listdir(scene_dir)):
        camera_path = os.path.join(scene_dir, camera_folder)
        if not os.path.isdir(camera_path):
            continue
        
        video_path = os.path.join(camera_path, 'video.mp4')
        if not os.path.exists(video_path):
            print(f"Video file not found in {camera_path}. Skipping.")
            continue
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        frame_interval = 1

        cam_output_dir = os.path.join(image_subset_dir, camera_folder)
        os.makedirs(cam_output_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = frame_id 
            frame_number_str = f"{frame_number:08d}"
            frame_path = os.path.join(cam_output_dir, f"{frame_number_str}.png")
            cv2.imwrite(frame_path, frame)
            
            frame_id += 1

        cap.release()
        print(f"Extracted frames for {camera_folder}")

import json
def create_annotation_positions(scene_dir, ground_gt_path, output_dir, grid_shape=(1400, 800), spacing=0.025):
    with open(ground_gt_path, 'r') as f:
        lines = f.readlines()

    num_cameras = 10

    annotations = {}
    
    xworld_min, xworld_max = -10, 10
    yworld_min, yworld_max = -15, 20

    # Calculate scale based on grid shape and world coordinate range
    x_scale = (xworld_max - xworld_min) / (grid_shape[1] * spacing)
    y_scale = (yworld_max - yworld_min) / (grid_shape[0] * spacing)

    for line in lines:
        camera_id, obj_id, frame_id, xmin, ymin, width, height, xworld, yworld = map(float, line.strip().split(' '))

        frame_id_str = f"{int(frame_id):08d}"

        if frame_id not in annotations:
            annotations[frame_id] = {}

        grid_x = int((xworld - xworld_min) / spacing)
        grid_y = int((yworld - yworld_min) / spacing)

        position_id = grid_y * grid_shape[0] + grid_x

        if obj_id not in annotations[frame_id]:
            annotations[frame_id][obj_id] = {
                "personID": int(obj_id),
                "positionID": position_id,
                "views": [{"viewNum": cam_id, "xmax": -1, "xmin": -1, "ymax": -1, "ymin": -1} for cam_id in range(num_cameras)]
            }

        # Update the current camera view with valid values
        annotations[frame_id][obj_id]["views"][int(camera_id)-1] = {
            "viewNum": int(camera_id)-1,
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


def main():
    scene_dir = "/data/aicity/scene_001"
    ground_gt_path = os.path.join(scene_dir, "ground_truth.txt")
    output_dir = os.path.join(scene_dir, "annotations_positions")
    extract_frames_per_camera(scene_dir, interval_sec=1)
    create_annotation_positions(scene_dir, ground_gt_path, output_dir)


if __name__ == "__main__":
    main()

