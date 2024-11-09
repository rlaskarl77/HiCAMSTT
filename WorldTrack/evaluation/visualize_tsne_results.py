import os
import os.path as osp
import torch
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import numpy as np
import kornia
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.manifold import TSNE


LOG_DIR = '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/baseline_tsne'
SAVE_DIR = '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/baseline_tsne/figures'

# LOG_DIR = '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_tsne'
# SAVE_DIR = '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_tsne/figures'


def load_feature(file, feat_dir):
    pid = int(file.split('_')[0])
    feat = np.load(osp.join(feat_dir, file))
    return feat, pid

def plot_tsne(log_dir=LOG_DIR, save_dir=SAVE_DIR, normalize=True):
    # load all features
    feat_dir = osp.join(log_dir, 'feat')
    features = []
    pids = []
    
    files = [file for file in os.listdir(feat_dir) if file.endswith('.npy')]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_feature, file, feat_dir) for file in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Loading features'):
            feat, pid = future.result()
            features.append(feat)
            pids.append(pid)
    
    features = np.array(features)
    pids = np.array(pids)
    
    if normalize:
        features = F.normalize(torch.from_numpy(features), dim=1).numpy()
    
    MAX_TRYOUT = 0 #5
    NUM_CHOICE = 10
    
    PERPLEXITY_LIST = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for i in range(MAX_TRYOUT):
        # choice random pid and their indexes for features and pids
        random_pid_list = np.random.choice(np.unique(pids), NUM_CHOICE, replace=False)
        random_indexes = []
        for pid in random_pid_list:
            random_indexes.extend(np.where(pids == pid)[0].tolist())
        random_features = features[random_indexes]
        random_pids = pids[random_indexes]
        
        for p in PERPLEXITY_LIST:
            tsne = TSNE(n_components=2, random_state=42, perplexity=p)
            tsne_results = tsne.fit_transform(random_features)
            
            print(f'Tryout {i}, Perplexity {p}: {random_pid_list}')
            
            visualize_tsne(tsne_results, random_pids, tag=f'tryout_{i}_perplexity_{p}', save_dir=save_dir)
        
    for p in PERPLEXITY_LIST:
        tsne = TSNE(n_components=2, random_state=42, perplexity=p)
        tsne_results = tsne.fit_transform(features)
        
        visualize_tsne(tsne_results, pids, tag=f'perplexity_{p}', save_dir=save_dir)


def visualize_tsne(tsne_results, pids, tag='', save_dir=SAVE_DIR):
    
    plt.figure(figsize=(10, 8))
    unique_pids = np.unique(pids)
    colors = plt.cm.get_cmap('tab20', len(unique_pids))
    pid_to_color = {pid: colors(i) for i, pid in enumerate(unique_pids)}
    pid_colors = [pid_to_color[pid] for pid in pids]
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=pid_colors, alpha=0.7)
    plt.title('t-SNE of Feature Vectors')
    legend_handles = [mpatches.Patch(color=mcolors.to_rgba(pid_to_color[pid]), label=f'{pid}') \
        for pid in unique_pids]
    plt.legend(handles=legend_handles, title='PID')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_name = osp.join(save_dir, f'tsne_{tag}.png') if tag else osp.join(save_dir, 'tsne.png')
        plt.savefig(fig_name)
    else:
        plt.show()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize t-SNE results')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='Path to log directory')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Path to save directory')
    parser.add_argument('--normalize', action='store_false', default=True, help='Normalize feature vectors')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    plot_tsne(args.log_dir, args.save_dir, args.normalize)

'''
command
base:
python visualize_tsne_results.py -u \
    --log_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/baseline_tsne' \
    --save_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/baseline_tsne/figures'
compare:
python visualize_tsne_results.py -u \
    --log_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_tsne' \
    --save_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_tsne/figures'
python visualize_tsne_results.py \
    --log_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tsne' \
    --save_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tsne/figures'
python visualize_tsne_results.py \
    --log_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_batch8_tsne' \
    --save_dir '/home/TrackTacular/experiments/lightning_logs/test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_batch8_tsne/figures'
python visualize_tsne_results.py \
    --log_dir '/home/TrackTacular/experiments/lightning_logs/test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tsne' \
    --save_dir '/home/TrackTacular/experiments/lightning_logs/test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tsne/figures'
'''