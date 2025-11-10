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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def save_pointcloud_as_ply(tensor, filename):
    """
    Save a Nx3 torch tensor as a .ply point cloud file.

    Args:
        tensor (torch.Tensor): A Nx3 tensor containing the point cloud data.
        filename (str): The name of the file to save the point cloud to.
    """
    if tensor.shape[1] != 3:
        raise ValueError("Tensor must have a shape of Nx3.")

    # Convert tensor to numpy array
    points = tensor.cpu().numpy()

    # Write the PLY file
    with open(filename, 'w') as ply_file:
        ply_file.write(f"ply\n")
        ply_file.write(f"format ascii 1.0\n")
        ply_file.write(f"element vertex {points.shape[0]}\n")
        ply_file.write(f"property float x\n")
        ply_file.write(f"property float y\n")
        ply_file.write(f"property float z\n")
        ply_file.write(f"end_header\n")
        
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")
    
def get_deformed_gaussian_centers(viewpoint_camera, pc : GaussianModel):
    means3D = pc.get_xyz
    
    opacity = pc._opacity
    shs = pc.get_features
    scales = pc._scaling
    rotations = pc._rotation
    if hasattr(pc,"_dynamic_xyz"):
        dynamic_gaussians_mask = pc._dynamic_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D[dynamic_gaussians_mask].shape[0],1)
        if dynamic_gaussians_mask.sum().item() > 1:
            # Perform deformation on the selected indices
            (
                means3D_deformed,
                scales_deformed,
                rotations_deformed,
                opacity_deformed,
                shs_deformed
            ) = pc._deformation(
                means3D[dynamic_gaussians_mask],
                scales[dynamic_gaussians_mask],
                rotations[dynamic_gaussians_mask],
                opacity[dynamic_gaussians_mask],
                shs[dynamic_gaussians_mask],
                time
            )
    return means3D_deformed




def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_opacity = None, stage = "fine", cam_type = None, time_dynamic_loss = False, out_of_frame_loss_flag = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    dynamic_movement_loss = None
    out_of_frame_stillness_loss = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if stage == "coarse" or "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif stage == "fine" or stage == "general_depth" or stage == "close_dynamic" or "train" in stage or "test" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
        if time_dynamic_loss:
            reference_time = max(viewpoint_camera.time-0.1,0.0)
            reference_time = torch.tensor(reference_time).to(means3D.device).repeat(means3D.shape[0],1)
            means3D_reference, scales_reference, rotations_reference, opacity_reference, shs_reference = pc._deformation(means3D, scales, 
                                                                    rotations, opacity, shs,
                                                                    reference_time)
            dynamic_movement_loss = abs(means3D_reference-means3D_final).mean()
            
        
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        shs_final = None
        colors_precomp = override_color

    if override_opacity is not None:
        assert(override_opacity.shape == opacity_final.shape)
        opacity_final = override_opacity

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # Compute a loss corresponding to things outside frame changing position, scale, rotation, opacity or color
    if out_of_frame_loss_flag and "fine" in stage:
        if len(pc.last_seen_means_scale_rot_opacity_shs)!=0 and pc.last_seen_means_scale_rot_opacity_shs["means3D_final"].shape == means3D_final.shape:
            out_of_frame_stillness_loss = pc.compute_masked_absolute_differences(means3D_final, scales_final, rotations_final, opacity_final, shs_final, ~(radii > 0))
        
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "dynamic_movement_loss": dynamic_movement_loss,
            "out_of_frame_stillness_loss": out_of_frame_stillness_loss
            }





def render_dynamic_compare(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0


    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if stage == "coarse" or "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif stage == "fine" or stage == "general_depth" or stage == "close_dynamic" or "train" in stage or "test" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)   
    else:
        raise NotImplementedError


    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)

    per_gaussian_valid_dynamic_movement = torch.zeros(means3D_final.shape[0])

    if hasattr(pc,"previous_visibility"):
        #continuous_visibility_filter = torch.tensor([(a and b).item() for a, b in zip(pc.previous_visibility.detach().cpu(), (radii > 0).detach().cpu())])
        #per_gaussian_valid_dynamic_movement = torch.sqrt(torch.sum(torch.pow(torch.subtract(means3D_final, pc.previous_positions), 2), dim=1)).detach().cpu()*continuous_visibility_filter
        # Keep everything on the GPU
        continuous_visibility_filter = (pc.previous_visibility & (radii > 0)).float()

        # Compute the per-Gaussian valid dynamic movement directly on GPU
        per_gaussian_valid_dynamic_movement = torch.sqrt(torch.sum((means3D_final - pc.previous_positions) ** 2, dim=1)) * continuous_visibility_filter

    pc.capture_visible_positions(means3D_final, radii > 0)

    return per_gaussian_valid_dynamic_movement


def render_with_dynamic_gaussians_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_opacity = None, stage = "fine", cam_type = None, training = False, render_normals = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if hasattr(pc,"_dynamic_xyz"):
        dynamic_gaussians_mask = pc._dynamic_xyz

    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )

        if dynamic_gaussians_mask.sum().item() > 1:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D[dynamic_gaussians_mask].shape[0],1)
        else:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features
    dynamic_movement_loss = None
    out_of_frame_stillness_loss = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation


    if hasattr(pc,"_dynamic_xyz") and dynamic_gaussians_mask.sum().item() > 1:

        # Perform deformation on the selected indices
        (
            means3D_deformed,
            scales_deformed,
            rotations_deformed,
            opacity_deformed,
            shs_deformed
        ) = pc._deformation(
            means3D[dynamic_gaussians_mask],
            scales[dynamic_gaussians_mask],
            rotations[dynamic_gaussians_mask],
            opacity[dynamic_gaussians_mask],
            shs[dynamic_gaussians_mask],
            time
        )

        # Initialize final tensors
        means3D_final = torch.empty_like(means3D)
        scales_final = torch.empty_like(scales)
        rotations_final = torch.empty_like(rotations)
        opacity_final = torch.empty_like(opacity)
        shs_final = torch.empty_like(shs)

        # Assign deformed values
        means3D_final[dynamic_gaussians_mask] = means3D_deformed
        scales_final[dynamic_gaussians_mask] = scales_deformed
        rotations_final[dynamic_gaussians_mask] = rotations_deformed
        opacity_final[dynamic_gaussians_mask] = opacity_deformed
        shs_final[dynamic_gaussians_mask] = shs_deformed

        # Assign original values to the rest
        means3D_final[~dynamic_gaussians_mask] = means3D[~dynamic_gaussians_mask]
        scales_final[~dynamic_gaussians_mask] = scales[~dynamic_gaussians_mask]
        rotations_final[~dynamic_gaussians_mask] = rotations[~dynamic_gaussians_mask]
        opacity_final[~dynamic_gaussians_mask] = opacity[~dynamic_gaussians_mask]
        shs_final[~dynamic_gaussians_mask] = shs[~dynamic_gaussians_mask]

    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs



    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        shs_final = None
        colors_precomp = override_color

    if override_opacity is not None:
        assert(override_opacity.shape == opacity_final.shape)
        opacity_final = override_opacity

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()

    # Initialize all possible returning variables to None
    rendered_image = None
    radii = None
    depth = None
    dynamic_only_splat = None
    dynamic_only_radii = None
    dynamic_only_depth = None
    static_only_splat = None
    static_only_radii = None
    static_only_depth = None

    # If we are training, we will render static and dynamic gaussians separately
    if training:
        # Check if there are any dynamic gaussians
        if dynamic_gaussians_mask.sum().item() > 1:

            if colors_precomp is not None:
                dynamic_masked_colors_precomp = colors_precomp[dynamic_gaussians_mask]
                non_dynamic_masked_colors_precomp = colors_precomp[~dynamic_gaussians_mask]
            else :
                dynamic_masked_colors_precomp = None
                non_dynamic_masked_colors_precomp = None

            if cov3D_precomp is not None:
                dynamic_masked_cov3D_precomp = cov3D_precomp[dynamic_gaussians_mask]
                non_dynamic_masked_cov3D_precomp = cov3D_precomp[~dynamic_gaussians_mask]
            else :
                dynamic_masked_cov3D_precomp = None
                non_dynamic_masked_cov3D_precomp = None

            dynamic_only_splat, dynamic_only_radii, dynamic_only_depth = rasterizer(
                means3D = means3D_final[dynamic_gaussians_mask],
                means2D = means2D[dynamic_gaussians_mask],
                shs = shs_final[dynamic_gaussians_mask],
                colors_precomp = dynamic_masked_colors_precomp,
                opacities = opacity_final[dynamic_gaussians_mask],
                scales = scales_final[dynamic_gaussians_mask],
                rotations = rotations_final[dynamic_gaussians_mask],
                cov3D_precomp = dynamic_masked_cov3D_precomp)
            
            static_only_splat, static_only_radii, static_only_depth = rasterizer(
                means3D = means3D_final[~dynamic_gaussians_mask],
                means2D = means2D[~dynamic_gaussians_mask],
                shs = shs_final[~dynamic_gaussians_mask],
                colors_precomp = non_dynamic_masked_colors_precomp,
                opacities = opacity_final[~dynamic_gaussians_mask],
                scales = scales_final[~dynamic_gaussians_mask],
                rotations = rotations_final[~dynamic_gaussians_mask],
                cov3D_precomp = non_dynamic_masked_cov3D_precomp)

            rendered_image, radii, depth = dynamic_only_splat, dynamic_only_radii, dynamic_only_depth

        # if there arent, the static only ones *are* the scene
        else:
            static_only_splat, static_only_radii, static_only_depth = rasterizer(
                means3D = means3D_final,
                means2D = means2D,
                shs = shs_final,
                colors_precomp = colors_precomp,
                opacities = opacity_final,
                scales = scales_final,
                rotations = rotations_final,
                cov3D_precomp = cov3D_precomp)
            rendered_image, radii, depth = static_only_splat, static_only_radii, static_only_depth

    # If we are not training or in the last fine stage, the entire scene is returned
    else:
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    # ============================================================================
    # Render normals if requested (use gsplat if available, fallback to PyTorch)
    # ============================================================================
    normal_map = None
    if render_normals:
        from time import time as get_time
        from utils.dn_splatter_utils import compute_gaussian_normals, render_normals
        
        t_normal_start = get_time()
        
        # Compute per-Gaussian normals from geometry
        normals_world = compute_gaussian_normals(
            quaternions=rotations_final,
            scales=scales_final,
            means3D=means3D_final,
            camera_center=viewpoint_camera.camera_center.cuda(),
            flip_to_camera=True
        )
        
        # Transform normals to camera space
        # NOTE: Normals are treated as passive "colors" by gsplat, not geometric entities
        # Therefore we must transform them to camera space ourselves
        R_w2c = viewpoint_camera.world_view_transform[:3, :3].cuda()
        normals_cam = (R_w2c @ normals_world.T).T  # [N, 3]
        normals_cam = torch.nn.functional.normalize(normals_cam, p=2, dim=-1)
        
        H = int(viewpoint_camera.image_height)
        W = int(viewpoint_camera.image_width)
        
        # gsplat viewmat expects world-to-cam transformation (COLMAP convention)
        # Our cameras store world_view_transform as TRANSPOSED, so transpose it back
        viewmat = viewpoint_camera.world_view_transform.T.cuda()  # [4, 4]
        
        # Construct intrinsic matrix K from FoV
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # fx = W / (2 * tan(FoVx/2))
        # fy = H / (2 * tan(FoVy/2))
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        fx = W / (2.0 * tanfovx)
        fy = H / (2.0 * tanfovy)
        cx = W / 2.0
        cy = H / 2.0
        
        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device="cuda")
        
        # Ensure opacities is 1D
        opacities_1d = opacity_final.squeeze(-1) if opacity_final.ndim > 1 else opacity_final
        
        # Compute depths for PyTorch fallback (if needed)
        cam_pos = viewpoint_camera.world_view_transform[3, :3].cuda()
        depths_for_fallback = torch.norm(means3D_final - cam_pos.unsqueeze(0), dim=-1)
        
        # Call unified render_normals (decides gsplat vs PyTorch internally)
        t_render_start = get_time()
        normal_map = render_normals(
            means3D=means3D_final,
            quats=rotations_final,
            scales=scales_final,
            opacities=opacities_1d,
            normals_cam=normals_cam,
            viewmat=viewmat,
            K=K,
            H=H,
            W=W,
            # PyTorch fallback parameters (only used if gsplat unavailable)
            means2D=screenspace_points,
            depths=depths_for_fallback,
            radii=radii
        )

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "static_only_render": static_only_splat,
            "static_only_radii": static_only_radii,
            "static_only_depth": static_only_depth,
            "dynamic_only_render": dynamic_only_splat,
            "dynamic_only_radii": dynamic_only_radii,
            "dynamic_only_depth": dynamic_only_depth,
            "dynamic_3D_means": means3D_final[dynamic_gaussians_mask],
            "normal_map": normal_map  # NEW: Rendered normal map [3, H, W] or None
            }


def render_dynamic_gaussians_mask_and_compare(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0


    # Set up rasterization configuration
    
    means3D = pc.get_xyz

    if hasattr(pc,"_dynamic_xyz"):
        dynamic_gaussians_mask = pc._dynamic_xyz
        
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        if dynamic_gaussians_mask.sum().item() > 1:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D[dynamic_gaussians_mask].shape[0],1)
        else:
            time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation


    if hasattr(pc,"_dynamic_xyz") and dynamic_gaussians_mask.sum().item() > 1:

        # Perform deformation on the selected indices
        (
            means3D_deformed,
            scales_deformed,
            rotations_deformed,
            opacity_deformed,
            shs_deformed
        ) = pc._deformation(
            means3D[dynamic_gaussians_mask],
            scales[dynamic_gaussians_mask],
            rotations[dynamic_gaussians_mask],
            opacity[dynamic_gaussians_mask],
            shs[dynamic_gaussians_mask],
            time
        )

        # Initialize final tensors
        means3D_final = torch.empty_like(means3D)
        scales_final = torch.empty_like(scales)
        rotations_final = torch.empty_like(rotations)
        opacity_final = torch.empty_like(opacity)
        shs_final = torch.empty_like(shs)

        # Assign deformed values
        means3D_final[dynamic_gaussians_mask] = means3D_deformed
        scales_final[dynamic_gaussians_mask] = scales_deformed
        rotations_final[dynamic_gaussians_mask] = rotations_deformed
        opacity_final[dynamic_gaussians_mask] = opacity_deformed
        shs_final[dynamic_gaussians_mask] = shs_deformed

        # Assign original values to the rest
        means3D_final[~dynamic_gaussians_mask] = means3D[~dynamic_gaussians_mask]
        scales_final[~dynamic_gaussians_mask] = scales[~dynamic_gaussians_mask]
        rotations_final[~dynamic_gaussians_mask] = rotations[~dynamic_gaussians_mask]
        opacity_final[~dynamic_gaussians_mask] = opacity[~dynamic_gaussians_mask]
        shs_final[~dynamic_gaussians_mask] = shs[~dynamic_gaussians_mask]

    else:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs


    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)

    per_gaussian_valid_dynamic_movement = torch.zeros(means3D_final.shape[0])

    if hasattr(pc,"previous_visibility"):
        #continuous_visibility_filter = torch.tensor([(a and b).item() for a, b in zip(pc.previous_visibility.detach().cpu(), (radii > 0).detach().cpu())])
        #per_gaussian_valid_dynamic_movement = torch.sqrt(torch.sum(torch.pow(torch.subtract(means3D_final, pc.previous_positions), 2), dim=1)).detach().cpu()*continuous_visibility_filter
        # Keep everything on the GPU
        continuous_visibility_filter = (pc.previous_visibility & (radii > 0)).float()

        # Compute the per-Gaussian valid dynamic movement directly on GPU
        per_gaussian_valid_dynamic_movement = torch.sqrt(torch.sum((means3D_final - pc.previous_positions) ** 2, dim=1)) * continuous_visibility_filter

    pc.capture_visible_positions(means3D_final, radii > 0)

    return per_gaussian_valid_dynamic_movement