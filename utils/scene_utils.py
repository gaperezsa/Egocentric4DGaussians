import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np

import copy
@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now, dataset_type, aria_rotated=False):
    """
    Render training images with 2-row layout:
    - Top row: GT RGB, GT Depth, GT Normal
    - Bottom row: Rendered RGB, Rendered Depth, Rendered Normal
    
    This layout makes vertical comparison easy for debugging.
    
    Args:
        aria_rotated: If True, rotate all images 90° CW for natural viewing of raw Aria orientation data
    """
    def render(gaussians, viewpoint, path, scaling, cam_type):
        # Get all rendered data
        if stage != "fine_coloring":
            render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type, training=True, render_normals=True)
        else:
            render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type, training=False, render_normals=True)
        
        # Labels
        label1 = f"stage:{stage}_train, iter:{iteration}"
        times = time_now / 60
        time_unit = "min" if times < 1 else "mins"
        label2 = f"time: {times:.2f} {time_unit}"
        
        # ====================================================================
        # Get Ground Truth Data
        # ====================================================================
        if dataset_type == "PanopticSports":
            gt_rgb = viewpoint['image'].permute(1, 2, 0).cpu().numpy()
        else:
            gt_rgb = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
        
        gt_depth = viewpoint.depth_image.cpu().unsqueeze(2).numpy()
        gt_depth_norm = gt_depth / (gt_depth.max() + 1e-6)
        gt_depth_norm = np.repeat(gt_depth_norm, 3, axis=2)
        
        # GT normal map (from camera if available)
        if hasattr(viewpoint, 'normal_map') and viewpoint.normal_map is not None:
            # normal_map is [3, H, W], convert to [H, W, 3]
            gt_normal = viewpoint.normal_map.permute(1, 2, 0).cpu().numpy()
            # Convert from [-1, 1] to [0, 1] for visualization
            gt_normal = (gt_normal + 1.0) / 2.0
        else:
            # Fallback: use black (no normal data)
            gt_normal = np.zeros_like(gt_rgb)
        
        # Rotate GT images 90° CW for Aria visualization (if raw Aria orientation)
        if aria_rotated:
            gt_rgb = np.rot90(gt_rgb, k=3)  # 90° CW
            gt_depth_norm = np.rot90(gt_depth_norm, k=3)
            gt_normal = np.rot90(gt_normal, k=3)
        
        # ====================================================================
        # Get Rendered Data
        # ====================================================================
        rendered_rgb = render_pkg["render"].permute(1, 2, 0).cpu().numpy()
        
        rendered_depth = render_pkg["depth"].permute(1, 2, 0).cpu().numpy()
        rendered_depth_norm = rendered_depth / (rendered_depth.max() + 1e-6)
        rendered_depth_norm = np.repeat(rendered_depth_norm, 3, axis=2)
        
        # Rendered normal map
        if "normal_map" in render_pkg and render_pkg["normal_map"] is not None:
            rendered_normal = render_pkg["normal_map"].permute(1, 2, 0).cpu().numpy()
            # Convert from [-1, 1] to [0, 1] for visualization
            rendered_normal = (rendered_normal + 1.0) / 2.0
        else:
            # Fallback: use black (no normal rendering)
            rendered_normal = np.zeros_like(rendered_rgb)
        
        # Rotate rendered images 90° CW for Aria visualization (if raw Aria orientation)
        if aria_rotated:
            rendered_rgb = np.rot90(rendered_rgb, k=3)  # 90° CW
            rendered_depth_norm = np.rot90(rendered_depth_norm, k=3)
            rendered_normal = np.rot90(rendered_normal, k=3)
        
        # ====================================================================
        # Create 2-row, 3-column layout
        # ====================================================================
        # Top row: GT RGB, GT Depth, GT Normal
        top_row = np.concatenate((gt_rgb, gt_depth_norm, gt_normal), axis=1)
        
        # Bottom row: Rendered RGB, Rendered Depth, Rendered Normal
        bottom_row = np.concatenate((rendered_rgb, rendered_depth_norm, rendered_normal), axis=1)
        
        # Combine rows
        combined_image = np.concatenate((top_row, bottom_row), axis=0)
        
        # Clip values and convert to 8-bit
        combined_image = np.clip(combined_image, 0, 1)
        image_with_labels = Image.fromarray((combined_image * 255).astype('uint8'))
        
        # ====================================================================
        # Add text labels
        # ====================================================================
        draw = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)
        text_color = (255, 0, 0)  # Red text
        
        # Stage/iteration label at top-left
        label1_position = (10, 10)
        draw.text(label1_position, label1, fill=text_color, font=font)
        
        # Time label at top-right
        label2_position = (image_with_labels.width - 250, 10)
        draw.text(label2_position, label2, fill=text_color, font=font)
        
        # Add column headers at the top
        small_font = ImageFont.truetype('./utils/TIMES.TTF', size=30)
        col_h = int(image_with_labels.width / 3)
        
        # Row headers on the left
        draw.text((10, int(image_with_labels.height / 2) - 40), "RENDERED", fill=(0, 255, 0), font=small_font)
        draw.text((10, 40), "GT", fill=(0, 255, 0), font=small_font)
        
        image_with_labels.save(path)
    
    # ====================================================================
    # Setup output directories
    # ====================================================================
    render_base_path = os.path.join(scene.model_path, f"{stage}_train_render")
    image_path = os.path.join(render_base_path, "images")
    # Removed pointcloud visualization to save time
    
    for path in [render_base_path, image_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # ====================================================================
    # Render for all viewpoints
    # ====================================================================
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path, f"{iteration:05}{idx:03}.jpg")
        render(gaussians, viewpoints[idx], image_save_path, scaling=1, cam_type=dataset_type)


def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    # 应用旋转和平移变换
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud.T)  # 转置点云数据以匹配Open3D的格式
    # transformed_point_cloud[2,:] = -transformed_point_cloud[2,:]
    # 可视化点云
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # 保存渲染结果为图片
    plt.savefig(filename)

@torch.no_grad()
def debug_render_training_image_by_mask(
    scene, gaussians, viewpoints,render_func, gaussian_mask,
    pipe, background, stage, iteration, time_now, dataset_type
):
    """
    Exactly like render_training_image(), but only splats the
    Gaussians for which gaussian_mask == True.
    """
    def render(viewpoint, out_path):
        # gaussian_mask: [N], pc._opacity is [N,1]
        # Build override_opacity: 1 for True, 0 for False
        override_opacity = gaussian_mask.to(gaussians._opacity.dtype).unsqueeze(1)   # [N,1]

        render_pkg = render_func(
            viewpoint, gaussians, pipe, background,
            stage=stage, cam_type=dataset_type,
            training=(stage != "fine_coloring"),
            override_opacity=override_opacity,
        )
        label1 = f"stage:{stage}_train_,iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        gt_depth = viewpoint.depth_image.cpu().unsqueeze(2).numpy()
        if iteration ==28:
            breakpoint()
        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
        
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)

        gt_depth_norm = gt_depth/gt_depth.max()
        gt_depth_norm = np.repeat(gt_depth_norm, 3, axis=2)
         
        # Aria rotated images fixing
        #gt_np = np.rot90(gt_np,k=3,axes=(0,1))
        #image_np = np.rot90(image_np,k=3,axes=(0,1))
        #depth_np = np.rot90(depth_np,k=3,axes=(0,1))


        image_np = np.concatenate((gt_np, image_np, depth_np,gt_depth_norm), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  # 转换为8位图像
        # 创建PIL图像对象的副本以绘制标签
        draw1 = ImageDraw.Draw(image_with_labels)

        # 选择字体和字体大小
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40)  # 请将路径替换为您选择的字体文件路径

        # 选择文本颜色
        text_color = (255, 0, 0)  # 白色

        # 选择标签的位置（左上角坐标）
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

        # 在图像上添加标签
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(out_path)

    render_base_path = os.path.join(scene.model_path, f"{stage}_train_render")
    mask_debug_path = os.path.join(render_base_path,"prune_debug")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_train_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(mask_debug_path):
        os.makedirs(mask_debug_path)
    # image:3,800,800

    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(mask_debug_path,f"prune_{iteration:05}{idx:03}.jpg")
        render(viewpoints[idx],image_save_path)


