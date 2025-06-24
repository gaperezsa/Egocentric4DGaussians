#!/usr/bin/env python3
"""
1) Decode align_rgb.mp4 + align_depth.avi → PNG frames
2) Reconstruct & ICP-align raw_pc.pcd into camera world frame
3) Run COLMAP adaptation (cameras.txt, images.txt, points3D.ply, resized images)
"""
import argparse, os, subprocess, shutil
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from copy import copy
from scipy.spatial.transform import Rotation
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
    files = sorted(f for f in os.listdir(folder) if f.endswith('.png'))
    arrs = []
    for f in files:
        im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Cannot read {f} in {folder}")
        arrs.append(im.astype(np.float32))
    return arrs  # in mm

def load_trajectory(log_path: str):
    traj = o3d.io.read_pinhole_camera_trajectory(log_path).parameters
    print(f"Loaded {len(traj)} camera poses from {log_path}")
    return traj

def make_intrinsic(npy_path: str, width: int, height: int):
    K = np.load(npy_path)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def reconstruct_scene(traj, intrinsic, rgb_dir, depth_dir, voxel_size):
    rgb_files  = natsorted(f for f in os.listdir(rgb_dir)   if f.endswith('.png'))
    depth_maps = load_sparse_depths(depth_dir)
    assert len(rgb_files) >= len(traj) and len(depth_maps) >= len(traj)
    all_pts = []
    for i, cam in enumerate(traj):
        color = o3d.io.read_image(os.path.join(rgb_dir,   rgb_files[i]))
        depth = depth_maps[i]  # mm
        depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, cam.extrinsic)
        pcd = pcd.voxel_down_sample(voxel_size)
        all_pts.append(np.asarray(pcd.points))
    recon = o3d.geometry.PointCloud()
    recon.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    recon, _ = recon.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.5)
    print("Reconstructed scene:", len(recon.points), "points")
    return recon

def run_icp(source, target, max_dist):
    print("Running ICP…")
    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"ICP fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.4f}")
    return reg.transformation

# ——— COLMAP adaptation helpers —————————————————————

def get_image_shape(path):
    if os.path.isfile(path):
        ext = path.lower()
        if ext.endswith(('.mp4','.avi','.mov','.mkv')):
            cap = cv2.VideoCapture(path)
            ret, f = cap.read()
            if not ret: raise RuntimeError(f"Cannot read {path}")
            h,w = f.shape[:2]; cap.release()
            return h,w
        else:
            w,h = Image.open(path).size
            return h,w
    elif os.path.isdir(path):
        imgs = sorted(p for p in os.listdir(path) if p.lower().endswith(('.png','.jpg')))
        if not imgs: raise RuntimeError(f"No images in {path}")
        w,h = Image.open(os.path.join(path, imgs[0])).size
        return h,w
    else:
        raise RuntimeError(f"{path} not found")

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
      [Rxx - Ryy - Rzz, 0, 0, 0],
      [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
      [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
      [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ])/3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3,0,1,2], np.argmax(vals)]
    if q[0]<0: q *= -1
    return q

def write_cameras_txt(out_folder, K, w, h):
    p = os.path.join(out_folder, "cameras.txt")
    with open(p,'w') as f:
        f.write("# Camera list…\n")
        f.write("#CAM_ID, MODEL, W, H, PARAMS\n")
        f.write(f"1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    print("Wrote", p)

def write_images_txt(out_folder, traj):
    p = os.path.join(out_folder, "images.txt")
    with open(p,'w') as f:
        f.write("# Image list…\n")
        f.write("#ID, QW, QX, QY, QZ, TX, TY, TZ, CAM_ID, NAME\n")
        for i,cam in enumerate(traj):
            P = copy(cam.extrinsic)
            R, t = P[:3,:3], P[:3,3]
            qw,qx,qy,qz = rotmat2qvec(R)
            name = f"camera_rgb_{i:05d}.jpg"
            f.write(f"{i} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n\n")
    print("Wrote", p)

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--raw_pcd",     required=True, help='Path to input .pcd point cloud')
    p.add_argument("--extrinsics",  required=True, help='Path to Open3D pinhole camera trajectory (.log)')
    p.add_argument("--intrinsics",  required=True, help='Path to .npy intrinsics file')
    p.add_argument("--rgb_video",   required=True, help='Path to origin image, folder of images, or .mp4 video')
    p.add_argument("--depth_video", required=True, help='Input sparse-depth AVI file')
    p.add_argument("--fps",         type=int, default=15)
    p.add_argument("--target",      help="path to sample image/video for RESOLUTION")
    p.add_argument("--target_width",  type=int, help="Either this or the target for resolution")
    p.add_argument("--target_height", type=int, help="Either this or the target for resolution")
    p.add_argument("--output",      required=True)
    p.add_argument("--voxel_size",  type=float, default=0.02)
    p.add_argument("--icp_dist",    type=float, default=1.0)
    args = p.parse_args()

    # 1) decode
    FRGB   = os.path.join(args.output, "rgb_frames")
    FDEPTH = os.path.join(args.output, "depth_frames")
    decode_video(args.rgb_video,   FRGB,   args.fps)
    decode_video(args.depth_video, FDEPTH, args.fps)

    # 2) align raw pcd
    traj  = load_trajectory(args.extrinsics)
    raw   = o3d.io.read_point_cloud(args.raw_pcd)
    raw,_ = raw.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.5)
    # intrinsic from first RGB
    samp = sorted(os.listdir(FRGB))[0]
    W,H  = Image.open(os.path.join(FRGB,samp)).size
    intrinsic = make_intrinsic(args.intrinsics, W, H)
    recon = reconstruct_scene(traj, intrinsic, FRGB, FDEPTH, args.voxel_size)
    M = run_icp(recon, raw, args.icp_dist)
    raw.transform(np.linalg.inv(M))
    aligned_pcd = os.path.join(args.output, "raw_pc_aligned.pcd")
    o3d.io.write_point_cloud(aligned_pcd, raw)
    print(" → Aligned PCD saved to", aligned_pcd)

    # 3) COLMAP‐adaptation
    SP0 = os.path.join(args.output, "sparse","0")
    os.makedirs(SP0, exist_ok=True)

    # resolutions
    origin_h, origin_w = H, W
    if args.target:
        target_h, target_w = get_image_shape(args.target)
    elif args.target_width and args.target_height:
        target_w, target_h = args.target_width, args.target_height
    else:
        p.error("Need --target or both --target_width/--target_height")

    # scale intrinsics
    K0 = np.load(args.intrinsics)
    K  = K0.copy()
    sx, sy = origin_w/target_w, origin_h/target_h
    K[0,:] /= sx;  K[1,:] /= sy;  K[2,:] = [0,0,1]

    write_cameras_txt(SP0, K, target_w, target_h)
    write_images_txt(SP0, traj)

    # write points3D.ply
    pts = np.asarray(raw.points)
    cnt = pts.shape[0]
    if cnt>200000:
        raw = raw.random_down_sample(200000/cnt)
        print(f"Downsampled from {cnt} → {len(raw.points)} pts")
    if not raw.has_normals():
        raw.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    colors = (np.asarray(raw.colors)*255).astype(np.uint8) if raw.has_colors() else np.zeros((len(raw.points),3),np.uint8)
    PLY = os.path.join(SP0,"points3D.ply")
    with open(PLY,'w') as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n"%len(raw.points))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\nend_header\n")
        for p, c, n in zip(raw.points, colors, raw.normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} {n[0]} {n[1]} {n[2]}\n")
    print("Wrote PLY →", PLY)

    # 4) resize & dump RGB images
    IMG_OUT = os.path.join(args.output, "images")
    os.makedirs(IMG_OUT, exist_ok=True)
    for i in range(len(traj)):
        src = os.path.join(FRGB, f"{i:05d}.png")
        im  = Image.open(src).resize((target_w,target_h), Image.LANCZOS)
        im.save(os.path.join(IMG_OUT, f"camera_rgb_{i:05d}.jpg"))
    print(f"Wrote {len(traj)} resized images →", IMG_OUT)

    shutil.rmtree(FRGB); shutil.rmtree(FDEPTH)

if __name__=="__main__":
    main()
