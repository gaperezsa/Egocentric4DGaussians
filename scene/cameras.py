#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2ViewAria

def write_point_cloud_to_ply(points: torch.Tensor, filename: str):
    """
    Writes a point cloud to a .ply file in ASCII format.
    
    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing world-space 3D points.
        filename (str): The filename (including path) where the .ply file will be saved.
    """
    # Ensure the tensor is on CPU and converted to a NumPy array
    if points.is_cuda:
        points = points.cpu()
    points_np = points.numpy()
    num_points = points_np.shape[0]

    with open(filename, 'w') as ply_file:
        # Write the PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        # Write each point's coordinates
        for point in points_np:
            ply_file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None, depth_image=None, dynamic_mask = None, normal_map = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        # breakpoint()
        # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)
        self.depth = depth
        self.depth_image = depth_image
        self.dynamic_mask = dynamic_mask
        self.normal_map = normal_map
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01
        
        # Cache image gradient for gradient-aware losses (computed once)
        self._image_gradient = None

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        #self.world_view_transform = torch.tensor(getWorld2ViewAria(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def get_image_gradient(self):
        """
        Compute and cache the image gradient magnitude for gradient-aware losses.
        This is computed once and reused across iterations.
        
        Returns:
            grad_mag: (H, W) tensor of gradient magnitudes
        """
        if self._image_gradient is None:
            from utils.dn_splatter_utils import compute_image_gradient
            # Compute gradient from RGB image
            self._image_gradient = compute_image_gradient(self.original_image)
        return self._image_gradient

    def backproject_mask_to_world(self):
        """
        Returns an (N,3) tensor of world‑space 3D points for all True pixels in camera.mask,
        using camera.depth and the camera intrinsics/extrinsics baked into the Camera object.
        """
        depth = self.depth_image        # shape (H, W)
        mask = self.dynamic_mask          # shape (H, W), dtype=torch.bool
        device = depth.device

        H, W = depth.shape

        # Compute focal lengths from FoV
        fx = W / (2 * torch.tan(torch.tensor(self.FoVx) / 2))
        fy = H / (2 * torch.tan(torch.tensor(self.FoVy) / 2))
        cx, cy = W * 0.5, H * 0.5

        # Get pixel indices where mask==True
        ys, xs = torch.nonzero(mask, as_tuple=True)
        zs = depth[ys, xs]

        # Convert pixel → camera coords
        x_cam = (xs.float() - cx) * zs / fx
        y_cam = (ys.float() - cy) * zs / fy
        ones = torch.ones_like(zs)

        cam_pts_h = torch.stack([x_cam, y_cam, zs, ones], dim=-1)  # (N,4)

        # Transform from camera to world: invert world_view_transform
        cam2world = torch.inverse(self.world_view_transform.to(device)).T
        # cam2world = self.world_view_transform.to(device)
        world_pts_h = (cam2world @ cam_pts_h.T).T

        return world_pts_h[:, :3]  # (N, 3)

    def to_device(self, device):
        self.original_image = self.original_image.to(device)
        if self.depth_image is not None:
            self.depth_image = self.depth_image.to(device)
        if self.dynamic_mask is not None:
            self.dynamic_mask = self.dynamic_mask.to(device)
        if self.normal_map is not None:
            self.normal_map = self.normal_map.to(device)

        # Move rotation and translation matrix
        #self.R = self.R.to(device)
        #self.T = self.T.to(device)
        # Also move precomputed transformation matrices:
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
    

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

