import argparse
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# =====================
# Helper functions
# =====================

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_next_bytes(fid, num_bytes, format_char_sequence):
    """
    Read the next num_bytes from fid and unpack them using the provided format.
    We assume little-endian byte order (as used by COLMAP).
    """
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


# =====================
# COLMAP file readers
# =====================

class Image:
    """
    Simple container for COLMAP image data.
    """
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids


def read_extrinsics_binary(path_to_model_file):
    """
    Read COLMAP image poses from a binary file.
    See COLMAP’s src/base/reconstruction.cc for details.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # read until ASCII null terminator
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            if num_points2D > 0:
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            else:
                xys = np.zeros((0, 2))
                point3D_ids = np.zeros((0,), dtype=int)
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_extrinsics_text(path):
    """
    Read COLMAP image poses from a text file.
    Taken (with minor modifications) from
    https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                # Next line contains the 2D point observations
                elems = fid.readline().split()
                if len(elems) >= 3:
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                           tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                else:
                    xys = np.zeros((0, 2))
                    point3D_ids = np.zeros((0,), dtype=int)
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


# =====================
# Build extrinsics using your method
# =====================

def images_to_extrinsics_myway(images):
    """
    Convert the images (a dict of COLMAP Image objects) into a numpy array
    of 4x4 camera extrinsic matrices using the direct approach:
        - R = qvec2rotmat(image.qvec)
        - T = image.tvec
    The extrinsics matrix is constructed as:
    
         [ R  t ]
         [ 0  1 ]
    """
    extrinsics = []
    for image in images.values():
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        extrinsics.append(M)
    return np.array(extrinsics)

def images_to_extrinsics(images):
    """
    Convert the images (a dict of COLMAP Image objects) into a numpy array
    of 4x4 camera extrinsic matrices for visualization.
    
    Note: In COLMAP, the stored rotation R and translation t (from qvec and tvec)
    are such that a 3D point X is transformed to camera coordinates by x = R*(X - C),
    where the camera center is C = -R.T * t.
    For visualization, we construct the camera-to-world extrinsics:
    
         [R^T   -R^T t]
         [0      1    ]
    """
    extrinsics = []
    for image in images.values():
        R = qvec2rotmat(image.qvec)  # this is R from world to camera
        R_cam2world = R.T
        C = -R_cam2world.dot(image.tvec)
        T = np.eye(4)
        T[:3, :3] = R_cam2world
        T[:3, 3] = C
        extrinsics.append(T)
    return np.array(extrinsics)


# =====================
# Visualization function (your original code)
# =====================

def plot_cameras(extrinsic_matrices):
    """
    Visualizes the extrinsics of multiple cameras in 3D space.
    
    Parameters:
    - extrinsic_matrices (numpy.ndarray): Array of camera extrinsics matrices (Nx4x4).
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    counter = 0
    for M in extrinsic_matrices:
        counter += 1
        if counter % 3 != 0:
            continue

        # Extract rotation and translation
        t = M[:3, 3]
        R = M[:3, :3]
        
        # Plot camera position
        ax.scatter(*t, marker='o', s=50)
        
        # Plot camera orientation axes
        origin = t
        colors = ['r', 'g', 'b']  # x, y, z axis colors
        for i in range(3):
            axis_direction = R[:, i]
            ax.quiver(*origin, *(axis_direction * 0.5), color=colors[i])

        # Additionally, emphasize the camera's forward direction (assumed -z in camera coords)
        z = -R[:, 2]
        #ax.quiver(*origin, *(z * 1.0), color='m', alpha=0.5, normalize=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Extrinsics Visualization')
    
    # Set all axes to range from _ to _
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    # Change the view angle: adjust elevation and azimuth
    ax.view_init(elev=0, azim=-90)  # Feel free to adjust these values
    
    # Show or save the plot
    # plt.show()
    plt.savefig("/home/gperezsantamaria/data/Egocentric4DGaussians/output/extrinsics_visualization.png")

# =====================
# Main routine
# =====================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP camera extrinsics from a .bin or .txt file using the direct approach."
    )
    parser.add_argument("extrinsics_file", help="Path to the COLMAP model file (images.bin or images.txt)")
    args = parser.parse_args()

    # Decide which reader to use based on file extension
    if args.extrinsics_file.endswith(".bin"):
        images = read_extrinsics_binary(args.extrinsics_file)
    elif args.extrinsics_file.endswith(".txt"):
        images = read_extrinsics_text(args.extrinsics_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .bin or .txt file.")

    extrinsics = images_to_extrinsics(images)
    plot_cameras(extrinsics)


if __name__ == "__main__":
    main()
