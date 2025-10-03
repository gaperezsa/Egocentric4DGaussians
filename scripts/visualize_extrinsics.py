import argparse
import struct
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# =====================
# Helper functions
# =====================

def qvec2rotmat(qvec):
    """
    Convert quaternion vector [qw, qx, qy, qz] to a 3x3 rotation matrix.
    """
    qw, qx, qy, qz = qvec
    # Based on standard formula
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

# =====================
# COLMAP readers
# =====================

def read_next_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    return struct.unpack('<' + fmt, data)

class Image:
    def __init__(self, id, qvec, tvec, camera_id, name):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name


def read_extrinsics_binary(path):
    images = {}
    with open(path, 'rb') as f:
        num = read_next_bytes(f, 8, 'Q')[0]
        for _ in range(num):
            vals = read_next_bytes(f, 64, 'idddddddi')
            image_id = vals[0]
            qvec = np.array(vals[1:5])
            tvec = np.array(vals[5:8])
            cam_id = vals[8]
            # read name
            name = ''
            while True:
                c = f.read(1)
                if c == b'': break
                if c == b'\x00': break
                name += c.decode('utf-8')
            # skip 2D points
            pts2d = read_next_bytes(f, 8, 'Q')[0]
            f.read(24 * pts2d)
            images[image_id] = Image(image_id, qvec, tvec, cam_id, name)
    return images


def read_extrinsics_text(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            elems = line.split()
            img_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            cam_id = int(elems[8])
            name = elems[9]
            images[img_id] = Image(img_id, qvec, tvec, cam_id, name)
            # skip next
            f.readline()
    return images

# =====================
# Extract camera centers
# =====================

def compute_camera_centers(images, binary=True):
    """
    Given dict of Image, return Nx3 array of camera centers in world coords.
    For binary/text, assume qvec/tvec meaning: x_cam = R*(X_world - C).
    So C = -R^T * tvec
    """
    centers = []
    for img in images.values():
        R_cam = qvec2rotmat(img.qvec)  # world->camera
        R_w2c = R_cam
        R_c2w = R_w2c.T
        C = -R_c2w.dot(img.tvec)
        centers.append(C)
    return np.stack(centers, axis=0)

# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser(
        description='Dump camera centers to PLY')
    parser.add_argument('--extrinsics', required=True,
                        help='Path to images.bin or images.txt')
    parser.add_argument('--output', required=True,
                        help='Output PLY file path')
    args = parser.parse_args()

    # Read
    if args.extrinsics.endswith('.bin'):
        imgs = read_extrinsics_binary(args.extrinsics)
    elif args.extrinsics.endswith('.txt'):
        imgs = read_extrinsics_text(args.extrinsics)
    else:
        raise ValueError('Unsupported format')

    # Compute centers
    centers = compute_camera_centers(imgs)

    # Create Open3D point cloud and write
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    o3d.io.write_point_cloud(args.output, pcd, write_ascii=True)
    print(f'Saved {len(centers)} camera centers to {args.output}')

if __name__ == '__main__':
    main()
