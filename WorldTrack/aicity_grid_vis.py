import pandas as pd
import matplotlib.pyplot as plt

file_path = '/data/aicity/scene_001/ground_truth.txt'

columns = ['camera_id', 'obj_id', 'frame_id', 'xmin', 'ymin', 'width', 'height', 'xworld', 'yworld']
df = pd.read_csv(file_path, sep=' ', names=columns)

def calculate_world_size(xworld_values, yworld_values):
    xworld_min, xworld_max = min(xworld_values), max(xworld_values)
    yworld_min, yworld_max = min(yworld_values), max(yworld_values)
    
    world_width = xworld_max - xworld_min + 1
    world_height = yworld_max - yworld_min + 1
    
    print(xworld_min, xworld_max)
    print(yworld_min, yworld_max)
    return world_width, world_height, xworld_min, yworld_min

def generate_position_id(xworld, yworld, world_width, xworld_min, yworld_min):
    return (yworld - yworld_min) * world_width + (xworld - xworld_min)

world_width, world_height, xworld_min, yworld_min = calculate_world_size(df['xworld'], df['yworld'])

df['positionID'] = df.apply(lambda row: generate_position_id(row['xworld'], row['yworld'], world_width, xworld_min, yworld_min), axis=1)

plt.figure(figsize=(10, 6))
plt.scatter(df['xworld'], df['yworld'], c='blue', marker='o', alpha=0.6, edgecolor='k')
plt.title('Scatter Plot of World Grid (xworld, yworld)')
plt.xlabel('xworld')
plt.ylabel('yworld')
plt.grid(True)

plot_path = 'world_grid_scatter_plot.png'
plt.savefig(plot_path)