#!/usr/bin/env python3
"""
Align HOI-4D raw_pc.pcd into the camera world frame
by ICP against a reconstruction from the align_rgb.mp4 + align_depth.avi.
"""
import argparse, os, subprocess
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from natsort import natsorted

def decode_video(video_path: str, output_dir: str, fps: int):
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-f image2 -start_number 0 -vf fps=fps={fps} '
        f'-qscale:v 2 "{output_dir}/%05d.png" -loglevel quiet'
    )
    print(f"Decoding {video_path} @ {fps}fps → {output_dir}")
    subprocess.run(cmd, shell=True, check=True)

def load_sparse_depths(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    arrs = []
    for f in files:
        im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read {f} from {folder}")
        arrs.append(im.astype(np.float32))  # still in mm
    return arrs

def load_trajectory(log_path):
    traj = o3d.io.read_pinhole_camera_trajectory(log_path).parameters
    print(f"Loaded {len(traj)} poses from {log_path}")
    return traj

def make_intrinsic(npy_path, width, height):
    K = np.load(npy_path)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def reconstruct_scene(traj, intrinsic, rgb_dir, depth_dir, voxel_size):
    rgb_files   = natsorted([f for f in os.listdir(rgb_dir)   if f.endswith('.png')])
    depth_maps  = load_sparse_depths(depth_dir)
    assert len(rgb_files) >= len(traj) and len(depth_maps) >= len(traj), \
        f"Need ≥{len(traj)} frames, got {len(rgb_files)} RGB and {len(depth_maps)} depth"
    all_pts = []
    for i, cam in enumerate(traj):
        color = o3d.io.read_image(os.path.join(rgb_dir,   rgb_files[i]))
        depth_np = depth_maps[i]           # in mm
        # convert to uint16 so Open3D knows it's millimeters
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_o3d,
            depth_scale=1000.0,       # mm → meters
            convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic, cam.extrinsic
        )
        pcd = pcd.voxel_down_sample(voxel_size)
        all_pts.append(np.asarray(pcd.points))
    recon = o3d.geometry.PointCloud()
    recon.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    recon, _ = recon.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.5)
    print("Reconstructed scene:", len(recon.points), "points")
    return recon

def run_icp(source, target, max_dist):
    print("Running ICP...")
    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"ICP fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")
    return reg.transformation

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--raw_pcd",     required=True, help="3DSeg/raw_pc.pcd")
    p.add_argument("--extrinsics",  required=True, help=".log trajectory")
    p.add_argument("--intrinsics",  required=True, help="intrin.npy")
    p.add_argument("--rgb_video",   required=True, help="align_rgb.mp4")
    p.add_argument("--depth_video", required=True, help="align_depth.avi")
    p.add_argument("--fps",         type=int, default=15, help="frame rate to extract")
    p.add_argument("--output",      required=True, help="output folder")
    p.add_argument("--voxel_size",  type=float, default=0.02, help="downsample voxel size")
    p.add_argument("--icp_dist",    type=float, default=1.0,  help="ICP max correspondence dist")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rgb_dir   = os.path.join(args.output, "rgb_frames")
    depth_dir = os.path.join(args.output, "depth_frames")

    # 1) ffmpeg decode both streams
    decode_video(args.rgb_video,   rgb_dir,   args.fps)
    decode_video(args.depth_video, depth_dir, args.fps)

    # 2) Load trajectory & raw PCD
    traj   = load_trajectory(args.extrinsics)
    raw_pc = o3d.io.read_point_cloud(args.raw_pcd)
    raw_pc, _ = raw_pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.5)

    # 3) Build intrinsic from first RGB
    sample = sorted(os.listdir(rgb_dir))[0]
    w,h = Image.open(os.path.join(rgb_dir, sample)).size
    intrinsic = make_intrinsic(args.intrinsics, w, h)

    # 4) Reconstruct scene
    recon_pc = reconstruct_scene(traj, intrinsic, rgb_dir, depth_dir, args.voxel_size)

    # 5) ICP: recon → raw
    mat = run_icp(recon_pc, raw_pc, args.icp_dist)

    # 6) Invert (so raw → world) & apply
    inv = np.linalg.inv(mat)
    raw_pc.transform(inv)

    # 7) Save aligned PCD
    out_pcd = os.path.join(args.output, "raw_pc_aligned.pcd")
    o3d.io.write_point_cloud(out_pcd, raw_pc)
    print("Wrote aligned PCD →", out_pcd)

if __name__ == "__main__":
    main()
