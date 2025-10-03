import os, sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from utils.general_utils import PILtoTorch
from gaussian_renderer import render_dynamic_compare
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def movement_by_rendering(model_path, name, views, gaussians, pipeline, background, cam_type, render_func = render_dynamic_compare):
    fig, ax = plt.subplots()
    print("point nums:",gaussians._xyz.shape[0])
    per_gaussians_dynamic_movement = torch.zeros(gaussians._xyz.shape[0])
    os.makedirs(os.path.join(model_path, name),exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            recent_movement = render_func(view, gaussians, pipeline, background,cam_type=cam_type).detach().cpu()
            per_gaussians_dynamic_movement = per_gaussians_dynamic_movement + recent_movement

        if idx % 500 == 0: 
            ax.plot(list(range(gaussians._xyz.shape[0])), per_gaussians_dynamic_movement)

            ax.set(xlabel='gaussian', ylabel='total movement (meters)',
                title='Per gaussian movement while in frame')
            ax.grid()

            fig.savefig(os.path.join(model_path, name, "in_frame_movement.png"))

            plt.cla()

            # Plot the histogram with log scale
            plt.hist(per_gaussians_dynamic_movement, bins=50)
            plt.yscale('log')  # Apply logarithmic scale to the y-axis
            plt.xlabel('Total Movement (meters)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Gaussian Movements (Log Scale)')

            # Save the histogram
            fig.savefig(os.path.join(model_path, name, "in_frame_movement_histogram.png"))

            # Clear the plot for the next one
            plt.cla()
    
    # Compute the 95th percentile value
    threshold = torch.quantile(per_gaussians_dynamic_movement, 0.95)
    
    # Clip values in the tensor to the 95th percentile
    per_gaussians_dynamic_movement = torch.clamp(per_gaussians_dynamic_movement, max=threshold)
    
    ax.plot(list(range(gaussians._xyz.shape[0])), per_gaussians_dynamic_movement)

    ax.set(xlabel='gaussian', ylabel='total movement (meters)',
        title='Per gaussian movement while in frame')
    ax.grid()

    fig.savefig(os.path.join(model_path, name, "in_frame_movement.png"))
    
    plt.cla()

    # Plot the histogram with log scale
    plt.hist(per_gaussians_dynamic_movement, bins=50)
    plt.yscale('log')  # Apply logarithmic scale to the y-axis
    plt.xlabel('Total Movement (meters)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Gaussian Movements (Log Scale)')

    # Save the histogram
    fig.savefig(os.path.join(model_path, name, "in_frame_movement_histogram.png"))

    # Clear the plot for the next one
    plt.cla()
    
    # Use plt to colorize gaussians by movement
    norm = plt.Normalize()
    colors = torch.tensor(plt.cm.jet(norm(per_gaussians_dynamic_movement))[:,:3]).type(torch.float).to(gaussians._xyz.device)

    #manual_normalized = per_gaussians_dynamic_movement/per_gaussians_dynamic_movement.max()
    #manual_normalized [ manual_normalized > 0.75] = 0
    #opacities = manual_normalized.unsqueeze(1).type(torch.float).to(gaussians._xyz.device)
    opacities = None
    
    torch.save(per_gaussians_dynamic_movement,os.path.join(model_path, name, "per_gaussians_dynamic_movement.pt"))
    return per_gaussians_dynamic_movement, colors, opacities

class SequentialPairsDataset(Dataset):
    def __init__(self, img_dir, depth_dir):
        self.img_dir_list = sorted(os.listdir(img_dir))
        self.depth_dir_list = sorted(os.listdir(depth_dir))
        self.len = len(os.listdir(self.img_dir))
    
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):

        depth_image = Image.open(self.depth_dir_list[idx])
        depth_image = torch.from_numpy((np.array(depth_image).astype(np.float32))/1000) #mm to meters

        image = Image.open(self.depth_dir_list[idx])
        image = PILtoTorch(image ,None)

        if idx+1< self.len:
            depth_image2 = Image.open(self.depth_dir_list[idx+1])
            depth_image2 = torch.from_numpy((np.array(depth_image2).astype(np.float32))/1000) #mm to meters

            
            image2 = Image.open(self.depth_dir_list[idx+1])
            image2 = PILtoTorch(image2 ,None)
        else:
            depth_image2 = None
            image2 = None
            
        
        return (depth_image, depth_image2), (image, image2)
    
class DynamicMaskProcessor():
    def __init__(self, images_dir, depth_dir):
        self.pairs_dataset = SequentialPairsDataset(images_dir, depth_dir)
        self.precomputed_flows = None
    def compute_flows(self):
        print('Computing optical flows...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the RAFT model
        flow_net = load_RAFT()  # You should implement this function to load your RAFT model
        flow_net = flow_net.to(device)
        flow_net.eval()
        
        # DataLoader
        dataloader = DataLoader(self.pairs_dataset, batch_size=12, shuffle=False, num_workers=4)
        
        flow_ij = []
        flow_ji = []
        
        with torch.no_grad():
            for ((depth_image1, depth_image2), (image1, image2)) in tqdm(dataloader):
                image1 = image1.to(device)  # Shape: [B, C, H, W]
                image2 = image2.to(device)  # Shape: [B, C, H, W]
                
                # RAFT expects inputs in [B, 3, H, W] and in range 0-255
                # Convert images from [0, 1] to [0, 255]
                image1_255 = image1 * 255.0
                image2_255 = image2 * 255.0
                
                # Compute flow from image1 to image2
                _, flow12 = flow_net(image1_255, image2_255, iters=20, test_mode=True)
                # Compute flow from image2 to image1
                _, flow21 = flow_net(image2_255, image1_255, iters=20, test_mode=True)
                
                flow_ij.append(flow12.cpu())
                flow_ji.append(flow21.cpu())
        
        # Concatenate the flows
        flow_ij = torch.cat(flow_ij, dim=0)
        flow_ji = torch.cat(flow_ji, dim=0)

        self.precomputed_flows = {'flow_ij': flow_ij, 'flow_ji': flow_ji}
        if flow_net is not None:
            del flow_net
        print('Optical flows computed.')
        



        
class DepthBasedWarping(torch.nn.Module):
    # tested
    def __init__(self) -> None:
        super().__init__()

    def get_relative_transform(src_R, src_t, tgt_R, tgt_t):
        tgt_R_inv = tgt_R.permute([0, 2, 1])
        relative_R = torch.matmul(tgt_R_inv, src_R)
        relative_t = torch.matmul(tgt_R_inv, src_t - tgt_t)
        return relative_R, relative_t

    def check_R_shape(self, R):
        r0, r1, r2 = R.shape
        assert r1 == 3 and r2 == 3

    def check_t_shape(self, t):
        t0, t1, t2 = t.shape
        assert t1 == 3 and t2 == 1

    def warp_by_disp(self, src_R, src_t, tgt_R, tgt_t, K, src_disp, coord, inv_K, debug_mode=False, use_depth=False):
        if debug_mode:
            B, C, H, W = src_disp.shape
            relative_R, relative_t = self.get_relative_transform(
                src_R, src_t, tgt_R, tgt_t)

            print(relative_t.shape)
            H_mat = K.matmul(relative_R.matmul(inv_K))  # Nx3x3
            flat_disp = src_disp.view([B, 1, H * W])  # Nx1xNpoints
            relative_t_flat = relative_t.expand([-1, -1, H*W])
            rot_coord = torch.matmul(H_mat, coord)
            tr_coord = flat_disp * \
                torch.matmul(K, relative_t_flat)
            tgt_coord = rot_coord + tr_coord
            normalization_factor = (tgt_coord[:, 2:, :] + 1e-6)
            rot_coord_normalized = rot_coord / normalization_factor
            tr_coord_normalized = tr_coord / normalization_factor
            tgt_coord_normalized = rot_coord_normalized + tr_coord_normalized
            debug_info = {}
            debug_info['tr_coord_normalized'] = tr_coord_normalized
            debug_info['rot_coord_normalized'] = rot_coord_normalized
            debug_info['tgt_coord_normalized'] = tgt_coord_normalized
            debug_info['tr_coord'] = tr_coord
            debug_info['rot_coord'] = rot_coord
            debug_info['normalization_factor'] = normalization_factor
            debug_info['relative_t_flat'] = relative_t_flat
            return (tgt_coord_normalized - coord).view([B, 3, H, W]), debug_info
        else:
            B, C, H, W = src_disp.shape
            relative_R, relative_t = self.get_relative_transform(
                src_R, src_t, tgt_R, tgt_t)
            H_mat = K.matmul(relative_R.matmul(inv_K))  # Nx3x3
            flat_disp = src_disp.view([B, 1, H * W])  # Nx1xNpoints
            if use_depth:
                tgt_coord = flat_disp * torch.matmul(H_mat, coord) + \
                    torch.matmul(K, relative_t)
            else:
                tgt_coord = torch.matmul(H_mat, coord) + flat_disp * \
                    torch.matmul(K, relative_t)
            tgt_coord = tgt_coord / (tgt_coord[:, -1:, :] + 1e-6)
            return (tgt_coord - coord).view([B, 3, H, W]), tgt_coord

    def generate_grid(self, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
        )
        self.coord = torch.ones(
            [1, 3, H, W], device=device, dtype=torch.float32)
        self.coord[0, 0, ...] = xx
        self.coord[0, 1, ...] = yy
        self.coord = self.coord.reshape([1, 3, H * W])
        self.jitted_warp_by_disp = None

    def forward(
        self,
        src_R,
        src_t,
        tgt_R,
        tgt_t,
        src_disp,
        K,
        inv_K,
        eps=1e-6,
        use_depth=False,
        check_shape=False,
        debug_mode=False,
    ):
        """warp the current depth frame and generate flow field.

        Args:
            src_R (FloatTensor): 1x3x3
            src_t (FloatTensor): 1x3x1
            tgt_R (FloatTensor): Nx3x3
            tgt_t (FloatTensor): Nx3x1
            src_disp (FloatTensor): Nx1XHxW
            src_K (FloatTensor): 1x3x3
        """
        if check_shape:
            self.check_R_shape(src_R)
            self.check_R_shape(tgt_R)
            self.check_t_shape(src_t)
            self.check_t_shape(tgt_t)

        _, _, H, W = src_disp.shape
        B = tgt_R.shape[0]
        device = src_disp.device
        if not hasattr(self, "coord"):
            self.generate_grid(H, W, device=device)
        else:
            if self.coord.shape[-1] != H * W:
                del self.coord
                self.generate_grid(H, W, device=device)

        return self.warp_by_disp(src_R, src_t, tgt_R, tgt_t, K, src_disp, self.coord, inv_K, debug_mode, use_depth)










def get_motion_mask_from_pairs(intrinsics_i, intrinsics_j, R_i, T_i, R_j, T_j, depth_maps_i, depth_maps_j, flow_ij, flow_ji):
    depth_wrapper = DepthBasedWarping()
    # Convert lists to tensors if necessary
    intrinsics_i = torch.stack(intrinsics_i).to(device)
    intrinsics_j = torch.stack(intrinsics_j).to(device)
    R_i = torch.stack(R_i).to(device)
    R_j = torch.stack(R_j).to(device)
    T_i = torch.stack(T_i).to(device)
    T_j = torch.stack(T_j).to(device)
    depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(device)
    depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(device)
    
    # Compute ego-motion induced optical flow
    ego_flow_1_2, _ = self.depth_wrapper(
        R_i, T_i, R_j, T_j, 1 / (depth_maps_i + 1e-6), intrinsics_j, torch.linalg.inv(intrinsics_i)
    )
    ego_flow_2_1, _ = self.depth_wrapper(
        R_j, T_j, R_i, T_i, 1 / (depth_maps_j + 1e-6), intrinsics_i, torch.linalg.inv(intrinsics_j)
    )

    # Compute error maps
    err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - flow_ij, dim=1)
    err_map_j = torch.norm(ego_flow_2_1[:, :2, ...] - flow_ji, dim=1)

    # Normalize error maps
    err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / \
                (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
    err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / \
                (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))

    # Threshold to get motion masks
    motion_mask_thre = 0.1  # Adjust threshold as needed
    dynamic_masks_i = err_map_i > motion_mask_thre
    dynamic_masks_j = err_map_j > motion_mask_thre

    return dynamic_masks_i, dynamic_masks_j
