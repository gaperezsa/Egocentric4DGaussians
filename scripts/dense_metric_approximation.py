import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Scale-and-shift depth maps with sparse GT.")
    parser.add_argument(
        "--relative_depth_folder", 
        type=str, 
        required=True, 
        help="Folder containing .npy files with dense relative depth (e.g. frame_0000.npy)."
    )
    parser.add_argument(
        "--sparse_depth_folder", 
        type=str, 
        required=True, 
        help="Folder containing .png files with sparse metric depth (e.g. 00000.png)."
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True, 
        help="Folder to write out the dense metric approximation .png images."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=500, 
        help="Number of iterations (epochs) for gradient descent."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1.0,  # Start at LR=1.0
        help="Initial learning rate for the optimizer."
    )
    return parser.parse_args()

def load_relative_depths(folder):
    """
    Loads all .npy files from the specified folder. 
    Returns a list of arrays sorted by filename.
    """
    npy_files = sorted([
        f for f in os.listdir(folder) 
        if f.endswith(".npy") and "frame_" in f
    ])
    data = []
    for f in npy_files:
        full_path = os.path.join(folder, f)
        arr = np.load(full_path)
        data.append(arr)
    return data

def load_sparse_depths(folder):
    """
    Loads all .png files from the specified folder. 
    Returns a list of arrays sorted by filename.
    """
    png_files = sorted([
        f for f in os.listdir(folder) 
        if f.endswith(".png")
    ])
    data = []
    for f in png_files:
        full_path = os.path.join(folder, f)
        # Read as a single-channel image
        sparse_img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        # Convert to float32 for safe handling
        sparse_img = sparse_img.astype(np.float32)
        data.append(sparse_img)
    return data

def main():
    args = parse_args()
    
    # ---------------------------
    # Device: CPU or GPU
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Make output directory if needed
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load data
    relative_depths = load_relative_depths(args.relative_depth_folder)  # list of arrays
    sparse_depths   = load_sparse_depths(args.sparse_depth_folder)      # list of arrays
    
    # We assume there's a 1-to-1 correspondence in sorted order:
    #   frame_0000.npy  <->  00000.png
    #   frame_0001.npy  <->  00001.png
    n_pairs = min(len(relative_depths), len(sparse_depths))
    
    # Collect all valid pixels into a single list for optimization
    X_list = []  # predicted relative depth
    Y_list = []  # sparse metric depth
    
    for i in range(n_pairs):
        rel_arr = relative_depths[i]
        sp_arr  = sparse_depths[i]
        
        # Check dimension matching; resize if necessary
        if rel_arr.shape != sp_arr.shape:
            print(f"Warning: shape mismatch for pair {i}. Resizing relative depth to match sparse depth.")
            
            target_height, target_width = sp_arr.shape[:2]
            rel_arr = cv2.resize(
                rel_arr,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Flatten
        rel_arr_flat = rel_arr.flatten()
        sp_arr_flat  = sp_arr.flatten()
        
        # Filter out invalid depths (where sp_arr_flat == 0)
        valid_mask = (sp_arr_flat > 0)
        X_list.append(rel_arr_flat[valid_mask])
        Y_list.append(sp_arr_flat[valid_mask])
    
    # Concatenate all valid pixels from all frames
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    
    print(f"Collected {len(X_all)} valid depth pixels in total.")
    
    # Convert to torch Tensors and move to device
    X_all_t = torch.from_numpy(X_all).float().to(device)
    Y_all_t = torch.from_numpy(Y_all).float().to(device)
    
    # Define learnable parameters: a, b, on the GPU if available
    a = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
    b = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))
    
    # Use an optimizer (Adam, etc.) with initial LR=1.0 (or from args)
    optimizer = optim.Adam([a, b], lr=args.learning_rate)
    
    # -------------------------------------------------
    # Linear LR Scheduler from LR to LR/100 over epochs
    # -------------------------------------------------
    # We'll go from start_factor=1.0 to end_factor=0.01 over total_iters = args.epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=1000
    )
    
    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Predicted = a*X + b
        pred = a * X_all_t + b
        
        # MSE loss
        loss = torch.mean((pred - Y_all_t) ** 2)
        
        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step(loss)  # update learning rate after each epoch
        
        # Print status every 50 epochs or on the first epoch
        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{args.epochs} "
                f"- Loss: {loss.item():.6f}, a={a.item():.4f}, b={b.item():.4f}, LR={current_lr:.6f}"
            )
    
    # Final parameters
    a_val = a.item()
    b_val = b.item()
    print(f"\nOptimization done. a = {a_val}, b = {b_val}")
    
    # Now apply the transformation to each dense relative map and save
    for i in range(n_pairs):
        rel_arr = relative_depths[i]
        
        # In case shape mismatch was fixed earlier, do the same resize again
        # because the original rel_arr might still be unresized in memory:
        sp_arr = sparse_depths[i]
        if rel_arr.shape != sp_arr.shape:
            target_height, target_width = sp_arr.shape[:2]
            rel_arr = cv2.resize(
                rel_arr,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )
        
        out_name = f"camera_depth_{i+1:05d}.png"
        out_path = os.path.join(args.output_folder, out_name)
        
        # Transform
        dense_metric = a_val * rel_arr + b_val
        
        # Clamp negative values to 0
        dense_metric = np.clip(dense_metric, 0, None)
        
        # Save as 16-bit PNG
        dense_metric_16 = np.round(dense_metric).astype(np.uint16)
        cv2.imwrite(out_path, dense_metric_16)
    
    print("All done! Dense metric depth images saved.")

if __name__ == "__main__":
    main()
