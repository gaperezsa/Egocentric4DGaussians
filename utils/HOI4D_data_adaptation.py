import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from copy import copy

##############################
# Utility functions
##############################

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def compute_sparse_point_cloud(image1, image2, K, extrinsic1, extrinsic2):
    """
    Compute a sparse 3D point cloud from two RGB frames using ORB feature matching and triangulation.
    
    Parameters:
      image1, image2: Grayscale images (numpy arrays).
      K: 3x3 camera intrinsics matrix.
      extrinsic1, extrinsic2: 4x4 extrinsic matrices (world -> camera) for the two frames.
    
    Returns:
      pts_3d_world: (N, 3) numpy array of triangulated 3D points in world coordinates.
    """
    # Increase number of features for denser matching.
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    
    # Use BFMatcher with knnMatch and ratio test.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    ratio_thresh = 0.75
    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    
    print("Found {} good matches".format(len(good_matches)))
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).T  # shape (2, N)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).T  # shape (2, N)
    
    # Compute projection matrices: P = K * [R|t]
    P1 = K @ extrinsic1[:3, :]
    P2 = K @ extrinsic2[:3, :]
    
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts_3d = (pts4d[:3, :] / pts4d[3, :]).T  # shape: (N, 3)
    
    # Filter points with positive depth and remove outliers.
    pts_3d = pts_3d[pts_3d[:, 2] > 0]
    pts_3d = pts_3d[pts_3d[:, 2] < np.percentile(pts_3d[:,2], 98)]
    
    # Transform points from camera (frame 1) to world coordinates.
    T_inv = np.linalg.inv(extrinsic1)
    ones = np.ones((pts_3d.shape[0], 1), dtype=pts_3d.dtype)
    pts_3d_h = np.hstack([pts_3d, ones])
    pts_3d_world_h = (T_inv @ pts_3d_h.T).T
    pts_3d_world = pts_3d_world_h[:, :3] / pts_3d_world_h[:, 3:4]
    
    return pts_3d_world

def save_point_cloud_to_ply(points, filename):
    """
    Save a point cloud (N, 3) numpy array to a PLY file in ASCII format,
    including dummy color and normal information.
    """
    num_points = points.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(num_points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        for p in points:
            f.write("{:.6f} {:.6f} {:.6f} {} {} {} {:.6f} {:.6f} {:.6f}\n".format(
                p[0], p[1], p[2],
                np.random.randint(1, high=255, dtype=int), np.random.randint(1, high=255, dtype=int), np.random.randint(1, high=255, dtype=int),   # random color
                0.0, 0.0, 1.0    # normal pointing in +Z direction
            ))

def write_cameras_txt(output_folder, K, width, height):
    """
    Write the cameras.txt file in COLMAP format.
    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    For PINHOLE: fx fy cx cy
    """
    cameras_txt_path = os.path.join(output_folder, "cameras.txt")
    with open(cameras_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE {} {} {} {} {} {}\n".format(
            width, height, K[0,0], K[1,1], K[0,2], K[1,2]
        ))
    print("Cameras file written to", cameras_txt_path)

def write_images_txt(output_folder, trajectory):
    """
    Write the images.txt file in COLMAP format.
    Each image has two lines:
      Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME
      Line 2: blank
    We invert the extrinsic to obtain the camera-to-world pose and then convert R to a quaternion.
    """
    images_txt_path = os.path.join(output_folder, "images.txt")
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        for i, cam in enumerate(trajectory):
            extrinsic = cam.extrinsic
            # Invert to get camera-to-world.
            #P = copy(np.linalg.inv(extrinsic))
            P = copy(extrinsic)
            R_mat = P[:3, :3]
            t = P[:3, 3]
            #quat = Rotation.from_matrix(R_mat).as_quat()  # returns [x, y, z, w]
            #qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            quat = rotmat2qvec(R_mat)
            qw, qx, qy, qz = quat
            image_name = f"camera_rgb_{i:05d}.jpg"
            f.write("{} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} 1 {}\n".format(
                i, qw, qx, qy, qz, t[0], t[1], t[2], image_name
            ))
            f.write("\n")
        breakpoint()
    print("Images file written to", images_txt_path)

##############################
# Main script
##############################

if __name__ == '__main__':
    # --- Input paths ---
    video_path = "/home/gperezsantamaria/data/HOI4D/HOI4D_release/ZY20210800004/H4/C1/N13/S194/s04/T4/align_rgb/image.mp4"
    intrinsics_path = "/home/gperezsantamaria/data/HOI4D/camera_params/ZY20210800004/intrin.npy"
    extrinsics_path = "/home/gperezsantamaria/data/HOI4D/annotations/HOI4D_annotations/ZY20210800004/H4/C1/N13/S194/s04/T4/3Dseg/output.log"
    output_folder = "/home/gperezsantamaria/data/Egocentric4DGaussians/data/HOI4D/Video2/adapted_colmap"

    os.makedirs(output_folder, exist_ok=True)

    # --- Load intrinsics ---
    original_K = np.load(intrinsics_path)
    # Suppose original_K are for full-res; we downscale by factor ~4 to target resolution 475x265.
    K = original_K / 4
    K[2,2] = 1
    target_width, target_height = 475, 265

    # --- Load extrinsics using Open3D ---
    trajectory = o3d.io.read_pinhole_camera_trajectory(extrinsics_path).parameters
    print("Loaded {} camera poses from extrinsics.".format(len(trajectory)))

    # --- Write cameras.txt and images.txt ---
    write_cameras_txt(output_folder, K, target_width, target_height)
    write_images_txt(output_folder, trajectory)

    # --- Extract frames from the video for point cloud initialization ---
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()  # Frame 0
    ret, frame1 = cap.read()  # Frame 1
    cap.release()
    if frame0 is None or frame1 is None:
        print("Error: Could not extract frames from the video.")
        exit(1)
    # Downscale frames to target resolution.
    frame0 = cv2.resize(frame0, (target_width, target_height))
    frame1 = cv2.resize(frame1, (target_width, target_height))
    image1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # --- For initialization, use the first two frames' extrinsics.
    extrinsic1 = trajectory[0].extrinsic
    extrinsic2 = trajectory[1].extrinsic
    
    pts_3d_world = compute_sparse_point_cloud(image1, image2, K, extrinsic1, extrinsic2)
    print("Triangulated {} 3D points from frames 0 and 1.".format(pts_3d_world.shape[0]))
    
    # --- Save the sparse point cloud (initialization) as a PLY file with colors and normals.
    ply_path = os.path.join(output_folder, "points3D.ply")
    save_point_cloud_to_ply(pts_3d_world, ply_path)
    print("Sparse point cloud saved to", ply_path)
