import argparse
import torch
import os
import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_scene(fname, pred_path, save_path):
    if not fname.endswith(".pth"):
        return

    scene_name = fname.replace(".pth", "")
    group = torch.load(os.path.join(pred_path, fname))

    if isinstance(group, torch.Tensor):
        group = group.cpu().numpy()

    if group is None or group.size == 0:
        print(f"Warning: {fname} is empty or 0D, skipping")
        return

    scene_folder = os.path.join(save_path, scene_name)
    os.makedirs(scene_folder, exist_ok=True)

    unique_instances = np.unique(group)
    unique_instances = unique_instances[unique_instances != -1]  # skip background

    pred_summary_lines = []
    for i, inst_id in enumerate(unique_instances):
        mask_file = f"{scene_name}_mask{i}.txt"
        mask_path = os.path.join(scene_folder, mask_file)
        mask = (group == inst_id).astype(np.int32).flatten()
        np.savetxt(mask_path, mask, fmt="%d")
        pred_summary_lines.append(f"{os.path.join(scene_name, mask_file)} 1 1.0")

    summary_file = os.path.join(save_path, f"{scene_name}.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(pred_summary_lines))

def convert_pth_to_txt(pred_path, save_path, num_workers=8):
    os.makedirs(save_path, exist_ok=True)
    files = [f for f in os.listdir(pred_path) if f.endswith(".pth")]

    # Create a partial function with fixed arguments
    worker_func = partial(process_scene, pred_path=pred_path, save_path=save_path)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm.tqdm(executor.map(worker_func, files),
                       total=len(files),
                       desc="Processing scenes"))

def get_args():
    parser = argparse.ArgumentParser(description='Segment Anything on ScanNet.')
    parser.add_argument('--pred_path', type=str, default='', help='path of pointcloud data')
    parser.add_argument('--save_path', type=str, help='where to save results')
    parser.add_argument('--workers', type=int, default=8, help='number of parallel workers')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Arguments:", args)
    convert_pth_to_txt(args.pred_path, args.save_path, args.workers)