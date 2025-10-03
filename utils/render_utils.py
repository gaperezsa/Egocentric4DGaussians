import torch
@torch.no_grad()
def get_state_at_time(pc,viewpoint_camera):    
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None

        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    # scales_final = pc.scaling_activation(scales_final)
    # rotations_final = pc.rotation_activation(rotations_final)
    # opacity = pc.opacity_activation(opacity_final)
    return means3D_final, scales_final, rotations_final, opacity, shs_final

def prune_by_visibility(gaussians, views, render_fn, pipeline, background, cam_type, threshold=0.1):
    """
    Render each view once (no saving), accumulate per-Gaussian visibility,
    then remove those seen in fewer than threshold fraction of frames.
    """
    device = background.device
    n_pts = gaussians.get_xyz.shape[0]
    # use int counter on GPU
    vis_counts = torch.zeros(n_pts, dtype=torch.int32, device=device)
    total = len(views)

    for view in views:
        view.to_device(device)
        with torch.no_grad():
            pkg = render_fn(
                view, gaussians, pipeline, background,
                cam_type=cam_type, training=False
            )
            # visibility_filter is a boolean mask of length n_pts
            vis_mask = pkg["visibility_filter"].to(vis_counts.dtype)
            vis_counts += vis_mask

    # compute frequency
    vis_freq = vis_counts.float() / float(total)
    # prune mask: True for those to remove
    prune_mask = vis_freq < threshold

    # actually remove them
    before = gaussians.get_xyz.shape[0]
    gaussians.prune_points(prune_mask)
    after = gaussians.get_xyz.shape[0]
    print(f"Pruned {before - after} / {before} Gaussians with visibility < {threshold*100:.0f}%")
    
def prune_by_average_radius(
    gaussians, views, render_fn, pipeline, background, cam_type,
    radius_thresh=1
):
    """
    For each Gaussian, sum its screen‐radius over all views,
    compute its average radius, and prune those below radius_thresh.
    """
    device = background.device
    n = gaussians.get_xyz.shape[0]
    radius_sum = torch.zeros(n, device=device)
    total = len(views)

    for v in views:
        v.to_device(device)
        with torch.no_grad():
            pkg = render_fn(
                v, gaussians, pipeline, background,
                cam_type=cam_type, training=False
            )
        # pkg["radii"] is a float tensor of length n (in pixels)
        radius_sum += pkg["radii"].to(device)

    avg_radius = radius_sum / float(total)
    prune_mask = avg_radius < radius_thresh

    before = n
    gaussians.prune_points(prune_mask)
    after = gaussians.get_xyz.shape[0]
    print(f"Pruned {before-after}/{before} Gaussians with avg radius < {radius_thresh}px")

def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Build a world→view (4×4) matrix that places the camera at `eye`,
    looks at `target`, with `up` vector = up.
    """
    # 1) camera z-axis = normalize(eye – target)
    z = eye - target
    z = z / z.norm()

    # 2) camera x-axis = normalize(up × z)
    x = torch.cross(up, z)
    x = x / x.norm()

    # 3) camera y-axis = z × x
    y = torch.cross(z, x)

    # 4) assemble rotation+translation
    R = torch.stack((x, y, z), dim=0)        # 3×3
    t = -R @ eye                            # 3
    M = torch.eye(4, device=eye.device)
    M[:3, :3] = R
    M[:3,  3] = t
    return M                                # world→view
