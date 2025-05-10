#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
from copy import copy
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_txt(output_folder, K, width, height):
    cameras_txt_path = os.path.join(output_folder, "cameras.txt")
    with open(cameras_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE {} {} {} {} {} {}\n".format(
            width, height, K[0,0], K[1,1], K[0,2], K[1,2]
        ))
    print("Cameras file written to", cameras_txt_path)


def write_images_txt(output_folder, trajectory):
    images_txt_path = os.path.join(output_folder, "images.txt")
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        for i, cam in enumerate(trajectory):
            extrinsic = cam.extrinsic
            P = copy(extrinsic)
            R_mat = P[:3, :3]
            t = P[:3, 3]
            quat = rotmat2qvec(R_mat)
            qw, qx, qy, qz = quat
            image_name = f"camera_rgb_{i:05d}.jpg"
            f.write("{} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} 1 {}\n".format(
                i, qw, qx, qy, qz, t[0], t[1], t[2], image_name
            ))
            f.write("\n")
    print("Images file written to", images_txt_path)


def get_image_shape(path):
    if os.path.isfile(path):
        ext = path.lower()
        if ext.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read frame from video {path}")
            h, w = frame.shape[:2]
            cap.release()
            return h, w
        else:
            img = Image.open(path)
            w, h = img.size
            return h, w
    elif os.path.isdir(path):
        imgs = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not imgs:
            raise RuntimeError(f"No images found in folder {path}")
        img = Image.open(os.path.join(path, imgs[0]))
        w, h = img.size
        return h, w
    else:
        raise RuntimeError(f"Reference path {path} not found")


def load_origin_frames(path):
    if os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return 'video', path
    elif os.path.isfile(path):
        return 'files', [path]
    elif os.path.isdir(path):
        imgs = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return 'files', [os.path.join(path, f) for f in imgs]
    else:
        raise RuntimeError(f"Origin path {path} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Adapt HOI4D intrinsics/extrinsics and PCD to COLMAP, "
                    "rescaling intrinsics to a target resolution and generating "
                    "resized RGB images."
    )
    parser.add_argument('--intrinsics', required=True, help='Path to .npy intrinsics file')
    parser.add_argument('--extrinsics', required=True, help='Path to Open3D pinhole camera trajectory (.log)')
    parser.add_argument('--pcd', required=True, help='Path to input .pcd point cloud')
    parser.add_argument('--origin', required=True, help='Path to origin image, folder of images, or .mp4 video')
    parser.add_argument('--target', help='Path to target image, folder of images, or .mp4 video in order to get the desired shape')
    parser.add_argument('--target_width', type=int, help='Target width if no --target provided')
    parser.add_argument('--target_height', type=int, help='Target height if no --target provided')
    parser.add_argument('--output', required=True, help='Output directory for COLMAP files and images')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "sparse", "0"), exist_ok=True)
    sparse_zero_output = os.path.join(args.output, "sparse", "0")

    # Determine target resolution
    if args.target:
        target_h, target_w = get_image_shape(args.target)
    elif args.target_width and args.target_height:
        target_w, target_h = args.target_width, args.target_height
    else:
        parser.error("Provide --target or both --target_width and --target_height")

    # Determine origin resolution
    origin_h, origin_w = get_image_shape(args.origin)

    # Compute scale factors
    sx = origin_w / target_w
    sy = origin_h / target_h

    # Load and rescale intrinsics
    original_K = np.load(args.intrinsics)
    K = original_K.copy()
    K[0, :] /= sx
    K[1, :] /= sy
    K[2, :] = [0, 0, 1]

    # Load trajectory
    traj = o3d.io.read_pinhole_camera_trajectory(args.extrinsics).parameters
    print(f"Loaded {len(traj)} camera poses from extrinsics.")

    # Write COLMAP files
    write_cameras_txt(sparse_zero_output, K, target_w, target_h)
    write_images_txt(sparse_zero_output, traj)

    # Read and simplify/convert PCD
    pcd = o3d.io.read_point_cloud(args.pcd)
    num_pts = np.asarray(pcd.points).shape[0]
    if num_pts > 200000:
        ratio = 200000 / num_pts
        print(f"Simplifying point cloud from {num_pts} to ~200000 points")
        pcd = pcd.random_down_sample(ratio)
    else:
        print(f"Point cloud has {num_pts} points; no simplification needed")

    # Estimate normals if missing
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else np.zeros((points.shape[0], 3), np.uint8)
    normals = np.asarray(pcd.normals)
    num_pts = points.shape[0]

    ply_out = os.path.join(sparse_zero_output, "points3D.ply")
    with open(ply_out, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_pts}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i in range(num_pts):
            x, y, z = points[i]
            r, g, b = colors[i]
            nx, ny, nz = normals[i]
            f.write(f"{x} {y} {z} {r} {g} {b} {nx} {ny} {nz}\n")
    print(f"Saved point cloud with normals ({num_pts} pts) as PLY at {ply_out}")

    # Generate resized RGB images
    images_out = os.path.join(args.output, "images")
    os.makedirs(images_out, exist_ok=True)
    mode, frames = load_origin_frames(args.origin)
    if mode == 'video':
        cap = cv2.VideoCapture(frames)
        for i in range(len(traj)):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Video ended before {len(traj)} frames")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((target_w, target_h), Image.LANCZOS)
            img.save(os.path.join(images_out, f"camera_rgb_{i:05d}.jpg"))
        cap.release()
    else:
        if len(frames) < len(traj):
            raise RuntimeError(f"Not enough origin frames ({len(frames)}) for {len(traj)} camera poses")
        for i in range(len(traj)):
            img = Image.open(frames[i])
            img = img.resize((target_w, target_h), Image.LANCZOS)
            img.save(os.path.join(images_out, f"camera_rgb_{i:05d}.jpg"))
    print(f"Resized and saved {len(traj)} images to {images_out}")

if __name__ == '__main__':
    main()
