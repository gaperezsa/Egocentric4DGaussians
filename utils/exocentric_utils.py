import numpy as np
from scipy.spatial.transform import Rotation as R


def read_extrinsics_text(path):
    """
    Parse a Colmap images.txt and return lists of camera centers and rotations.
    Returns:
      C_list: list of shape-(3,) numpy arrays (world-space camera centers)
      R_list: list of shape-(3,3) numpy arrays (camera->world rotations)
    """
    C_list, R_list = [], []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            elems = line.split()
            # parse quaternion and translation
            idx = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))  # [qw,qx,qy,qz]
            tvec = np.array(list(map(float, elems[5:8])))  # [Tx,Ty,Tz]
            # skip the 2nd line of points
            _ = f.readline()

            # build camera->world rotation R_c2w
            # scipy expects [x,y,z,w]
            quat_xyzw = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
            R_w2c = R.from_quat(quat_xyzw).as_matrix()
            R_c2w = R_w2c.T

            # compute camera center C = -R_c2w @ tvec
            C = -R_c2w.dot(tvec)

            C_list.append(C)
            R_list.append(R_c2w.T)
    return C_list, R_list


def compute_average_and_zoom(C_list, R_list, distance=1.0, up=(0,1,0)):
    """
    Compute a single exocentric camera by averaging centers
    and forward directions, then backing off by `distance`.

    Args:
      C_list: list of N camera centers, each shape (3,)
      R_list: list of N rotations (camera->world), each shape (3,3)
      distance: float, how far to step back
      up: world up vector
    Returns:
      R_new: (3,3) camera->world rotation
      T_new: (3,) world->camera translation
    """
    # average center
    C_avg = np.mean(np.stack(C_list, axis=0), axis=0)
    # average forward axes (world) = R_i @ [0,0,1]
    fwds = np.stack([R.dot([0,0,1]) for R in R_list], axis=0)
    f_avg = fwds.mean(axis=0)
    f_avg /= np.linalg.norm(f_avg)

    # new center: back off along -f_avg
    C_new = C_avg - distance * f_avg

    # build look-at basis: fwd->target, right, up
    target = C_avg
    up_vec = np.array(up, dtype=float)
    fwd = (target - C_new)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up_vec, fwd)
    right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    true_up /= np.linalg.norm(true_up)

    # camera->world basis columns [right, up, fwd]
    R_c2w_new = np.stack([right, true_up, fwd], axis=1)
    # world->camera
    R_w2c_new = R_c2w_new.T

    # translation t = -R_w2c @ C_new
    T_new = -R_w2c_new.dot(C_new)
    return R_c2w_new, T_new


def compute_exocentric_from_file(path, distance=0.03, up=(0,1,0)):
    """
    High-level util: read extrinsics.txt, compute exocentric R and T.
    Returns R_new, T_new.
    """
    C_list, R_list = read_extrinsics_text(path)
    return compute_average_and_zoom(C_list, R_list, distance=distance, up=up)
