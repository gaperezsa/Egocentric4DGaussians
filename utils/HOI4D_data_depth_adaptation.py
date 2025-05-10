#!/usr/bin/env python3
import os
import subprocess
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
from PIL import Image
from tqdm import tqdm

# ------------------------------
# STEP 1: Decode sparse depth AVI → PNG frames
# ------------------------------
def decode_video(video_path: str, output_dir: str, fps: int):
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-f image2 -start_number 0 -vf fps=fps={fps} '
        f'-qscale:v 2 "{output_dir}/%05d.png" -loglevel quiet'
    )
    print(f"Decoding video at {fps} fps → {output_dir}")
    subprocess.run(cmd, shell=True, check=True)

# ------------------------------
# STEP 2: Resize sparse frames to match RGB resolution
# ------------------------------
def resize_depth_images(depth_dir: str, rgb_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # pick a sample RGB to get target size
    samples = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png','.jpg'))]
    assert samples, f"No RGB found in {rgb_dir}"
    w, h = Image.open(os.path.join(rgb_dir, samples[0])).size
    print(f"Resizing depth frames from {depth_dir} to {w}×{h} → {output_dir}")
    for fn in sorted(os.listdir(depth_dir)):
        if not fn.endswith('.png'): continue
        im = Image.open(os.path.join(depth_dir, fn))
        im = im.resize((w, h), Image.NEAREST)
        im.save(os.path.join(output_dir, fn))

# ------------------------------
# STEP 2.1:  Resize dense non-metric (relative) depths
# ------------------------------

def resize_relative_depths(input_folder: str, rgb_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # get target size from an RGB sample
    samples = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png','.jpg'))]
    assert samples, f"No RGB found in {rgb_dir}"
    w, h = Image.open(os.path.join(rgb_dir, samples[0])).size
    print(f"Resizing relative depth arrays from {input_folder} to {w}×{h} → {output_dir}")
    for fn in sorted(os.listdir(input_folder)):
        if fn.endswith('.npy') and fn.startswith('frame'):
            arr = np.load(os.path.join(input_folder, fn)).astype(np.float32)
            # cv2.resize takes (width, height)
            arr_resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(output_dir, fn), arr_resized)

# ------------------------------
# STEP 3: Load relative & sparse depths
# ------------------------------
def load_relative_depths(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy') and f.startswith('frame_')])
    return [np.load(os.path.join(folder, f)) for f in files]

def load_sparse_depths(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    arrs = []
    for f in files:
        im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        arrs.append(im)
    return arrs

# ------------------------------
# STEP 4: Optimize scale (a,b)
# ------------------------------
def optimize_scale_shift(X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, device):
    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)
    a = nn.Parameter(torch.tensor([1.0], device=device))
    b = nn.Parameter(torch.tensor([0.0], device=device))
    opt = optim.Adam([a, b], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        patience=100
    )
    for ep in range(epochs):
        opt.zero_grad()
        pred = a * X_t + b
        loss = torch.mean((pred - Y_t) ** 2)
        loss.backward()
        opt.step()
        scheduler.step(loss)
        if ep%1000 == 0:
            print(f"ep: {ep}, lr: {opt.param_groups[0]['lr']}, loss: {loss.item()}")
    return a.item(), b.item()

# ------------------------------
# STEP 5: Apply transform & save dense metric maps
# ------------------------------
def apply_and_save(relative_list, a, b, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    for i, rel in enumerate(relative_list):
        dense = a * rel + b
        dense = np.clip(dense, 0, None)
        out = os.path.join(out_folder, f"camera_depth_{i:05d}.png")
        cv2.imwrite(out, dense.round().astype(np.uint16))

# ------------------------------
# Main pipeline
# ------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Full pipeline: sparse AVI → dense metric depth maps"
    )
    p.add_argument('--sparse_avi', required=True, help='Input sparse-depth AVI file')
    p.add_argument('--rgb_folder', required=True, help='Folder of RGB images for target size')
    p.add_argument('--relative_folder', required=True, help='Folder of .npy relative-depth outputs')
    p.add_argument('--output_folder', required=True, help='Where final .png depth maps go')
    p.add_argument('--fps', type=int, default=15, help='FPS for frame extraction')
    p.add_argument('--epochs', type=int, default=5000, help='Epochs for scale/shift opt')
    p.add_argument('--learning_rate', type=float, default=10.0, help='LR for optimizer')
    args = p.parse_args()
    
    raw_dir = os.path.join(args.output_folder, 'raw_depth_frames')
    resized_dir = os.path.join(args.output_folder, 'resized_depth_frames')
    final_dir = os.path.join(args.output_folder, 'depth')
    resized_rel_dir = os.path.join(args.output_folder, 'resized_relative_depths')

    # 1) Decode
    decode_video(args.sparse_avi, raw_dir, args.fps)

    # 2) Resize
    resize_depth_images(raw_dir, args.rgb_folder, resized_dir)

    # 2.1) Resize relative (dense non-metric) depths
    resize_relative_depths(args.relative_folder, args.rgb_folder, resized_rel_dir)

    # 3) Load
    rels = load_relative_depths(resized_rel_dir)
    spars = load_sparse_depths(resized_dir)
    n = min(len(rels), len(spars))
    X_list, Y_list = [], []
    for i in range(n):
        r, s = rels[i], spars[i]
        if r.shape != s.shape:
            s_h, s_w = s.shape[:2]
            r = cv2.resize(r, (s_w, s_h), interpolation=cv2.INTER_NEAREST)
        rf, sf = r.flatten(), s.flatten()
        mask = (sf > 0)
        X_list.append(rf[mask]); Y_list.append(sf[mask])
    X_all = np.concatenate(X_list)
    Y_all = np.concatenate(Y_list)
    print(f"Collected {len(X_all)} valid points for optimization.")

    # 4) Optimize a,b
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a, b = optimize_scale_shift(X_all, Y_all, args.epochs, args.learning_rate, device)
    print(f"Learned a={a:.4f}, b={b:.4f}")

    # 5) Apply & save
    apply_and_save(rels[:n], a, b, final_dir)
    print(f"Dense metric maps written to {final_dir}")

if __name__ == '__main__':
    main()
