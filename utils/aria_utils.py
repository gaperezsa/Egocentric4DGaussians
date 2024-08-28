import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
import numpy as np
import open3d as o3d
import random
import shutil
import csv

import cv2
import numpy as np
import os

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from tqdm import tqdm

def magnify_image(image, scale=1.0):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    magnified_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return magnified_image

def create_data_provider(vrsfile):
    print(f"Creating data provider from {vrsfile}")
    provider = data_provider.create_vrs_data_provider(vrsfile)
    if not provider:
        print("Invalid vrs data provider")
        return None
    return provider




def undistort_depth_fisheye_image(input_path, output_path, provider,pinhole_width = 1408,pinhole_height=1408,focal_length=610.9410078676575):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib("camera-rgb")
    #dst_calib = calibration.get_linear_camera_calibration(1024, 1024, 300, "camera-rgb")
    dst_calib = calibration.get_linear_camera_calibration(pinhole_width, pinhole_height, focal_length, "camera-rgb")

    # Read image
    #numpy_array = np.load(input_path)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]

    # Undistort image
    rectified_array = calibration.distort_depth_by_calibration(img, dst_calib, src_calib)
    # Optionally magnify or crop the image to reduce black borders
    # rectified_array = magnify_image(rectified_array, scale=1.1)  # Assuming a magnify_image function exists

    # Save the corrected image
    cv2.imwrite(output_path, rectified_array)

def undistort_fisheye_image(input_path, output_path, provider,pinhole_width = 1408,pinhole_height=1408,focal_length=610.9410078676575):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib("camera-rgb")
    #dst_calib = calibration.get_linear_camera_calibration(1024, 1024, 300, "camera-rgb")
    dst_calib = calibration.get_linear_camera_calibration(pinhole_width, pinhole_height, focal_length, "camera-rgb")

    # Read image
    img = cv2.imread(input_path)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]

    # Undistort image
    rectified_array = calibration.distort_by_calibration(img, dst_calib, src_calib, InterpolationMethod.BILINEAR)

    # Optionally magnify or crop the image to reduce black borders
    # rectified_array = magnify_image(rectified_array, scale=1.1)  # Assuming a magnify_image function exists

    # Save the corrected image
    cv2.imwrite(output_path, rectified_array)

def undistort_fisheye_segmentation(input_path, output_path, provider,pinhole_width = 1408,pinhole_height=1408,focal_length=610.9410078676575):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib("camera-rgb")
    #dst_calib = calibration.get_linear_camera_calibration(1024, 1024, 300, "camera-rgb")
    dst_calib = calibration.get_linear_camera_calibration(pinhole_width, pinhole_height, focal_length, "camera-rgb")

    # Read image
    img = np.load(input_path)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]

    # Undistort image
    rectified_array = calibration.distort_by_calibration(img, dst_calib, src_calib, InterpolationMethod.NEAREST_NEIGHBOR)

    # Optionally magnify or crop the image to reduce black borders
    # rectified_array = magnify_image(rectified_array, scale=1.1)  # Assuming a magnify_image function exists

    # Save the corrected image
    np.save(output_path, rectified_array)

def process_aria_fisheye(input_folder, output_folder, vrsfile, pinhole_width, pinhole_height, focal_length):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    provider = create_data_provider(vrsfile)

    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.lower().startswith('camera_depth'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            undistort_depth_fisheye_image(input_path, output_path, provider,pinhole_width,pinhole_height,focal_length)
        elif file_name.lower().startswith('camera_rgb'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            undistort_fisheye_image(input_path, output_path, provider,pinhole_width,pinhole_height,focal_length)
        elif file_name.lower().startswith('camera_segmentation'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            undistort_fisheye_segmentation(input_path, output_path, provider,pinhole_width,pinhole_height,focal_length)
        elif file_name.lower().startswith('bounding_box'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            undistort_fisheye_segmentation(input_path, output_path, provider,pinhole_width,pinhole_height,focal_length)
        


def read_images_id_mapping(images_txt_path):
    """Reads the images.txt file and creates a mapping from filename to COLMAP image ID."""
    mapping = {}
    with open(images_txt_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) > 9:  # This line contains image metadata
                image_id = int(parts[0])
                file_path = parts[9]
                mapping[file_path] = image_id
    return mapping

def convert_transforms(input_json_path, images_txt_path, output_json_path, input_ply_path, pinhole_width, pinhole_height, focal_length):
    # Read the original ARIA JSON
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Read the image ID mappings
    image_id_mapping = read_images_id_mapping(images_txt_path)

    # Prepare the new JSON structure
    '''
    output_data = {
        "w": data['frames'][0]['w'],
        "h": data['frames'][0]['h'],
        "fl_x": data['frames'][0]['fl_x'],
        "fl_y": data['frames'][0]['fl_y'],
        "cx": data['frames'][0]['cx'],
        "cy": data['frames'][0]['cy'],
        "k1": data['frames'][0]['distortion_params'][0],
        "k2": data['frames'][0]['distortion_params'][1],
        "p1": data['frames'][0]['distortion_params'][6],
        "p2": data['frames'][0]['distortion_params'][7],
        "k3": data['frames'][0]['distortion_params'][2],
        "k4": data['frames'][0]['distortion_params'][3],
        "s0": data['frames'][0]['distortion_params'][8],
        "s2": data['frames'][0]['distortion_params'][10],
        "camera_model": "THIN_PRISM_FISHEYE",
        "frames": [],
        "applied_transform": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
        "ply_file_path": os.path.basename(input_ply_path)
    }
    '''
    # Prepare camera instrinsics structure
    if pinhole_width is None:
        cx = data['frames'][0]['cx']
    else:
        cx = pinhole_width/2

    if pinhole_height is None:
        cy = data['frames'][0]['cy']
    else:
        cy = pinhole_height/2

    output_data = {
        "w": pinhole_width,
        "h": pinhole_height,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": cx,
        "cy": cy,
        "camera_model": "PINHOLE",
        "frames": [],
        "applied_transform": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
        "ply_file_path": os.path.basename(input_ply_path)
    }

    # Process each frame
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        # Applying the axis swap: swap X and Z axes
        # Create a swap matrix
        rotation_matrix = transform[:3, :3]
        translation_vector = transform[:3, 3]

        # Apply 180 degree rotation about the x-axis to the rotation part
        #rotation_180_x = np.array([ [1, 0, 0],
                                    #[0, -1, 0],
                                    #[0, 0, -1]])
        #rotation_matrix = rotation_matrix.dot(rotation_180_x)

        # Swap x and z in the translation vector
        # translation_vector = [translation_vector[2], translation_vector[1], translation_vector[0]]

        # transforming back to homogeneus coord
        # transformed = np.zeros((4,4))
        # transformed[:3,:3] = rotation_matrix
        # transformed[:3, 3] = translation_vector
        # transformed[-1,-1] = 1
        colmap_frame = {
            "file_path": "images/" + os.path.basename(frame['file_path']),
            "transform_matrix": transform.tolist(),
            "colmap_im_id": image_id_mapping.get(os.path.basename(frame['file_path']), -1)
        }
        output_data['frames'].append(colmap_frame)

    # Write the new JSON file
    with open(output_json_path, 'w') as file:
        json.dump(output_data, file, indent=4)

def organize_images_for_colmap(base_path):
    # Define the paths for 'colmap' directories and 'images' directories
    colmap_path = os.path.join(base_path, 'colmap')
    images_path_colmap = os.path.join(colmap_path, 'images')
    images_path_base = os.path.join(base_path, 'images')
    depth_path_base = os.path.join(base_path, 'depth')
    segmentation_path_base = os.path.join(base_path, 'segmentation')
    bounding_boxes_path_base = os.path.join(base_path, 'bounding_boxes')

    # Ensure the 'colmap' directory and both 'images' directories exist
    for path in [colmap_path, images_path_colmap, images_path_base, depth_path_base, segmentation_path_base, bounding_boxes_path_base]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # Find and copy (not move to keep in both locations) all .jpg files to the 'images' directories
    for filename in os.listdir(base_path):
        if filename.lower().startswith('camera_rgb'):
            src_path = os.path.join(base_path, filename)
            dst_path_colmap = os.path.join(images_path_colmap, filename)
            dst_path_base = os.path.join(images_path_base, filename)
            shutil.copy(src_path, dst_path_colmap)
            shutil.move(src_path, dst_path_base)
            print(f"Copied {filename} to {images_path_colmap} and {images_path_base}")
        elif filename.lower().startswith('camera_depth'):
            src_path = os.path.join(base_path, filename)
            dst_path_base = os.path.join(depth_path_base, filename)
            shutil.move(src_path, dst_path_base)
            print(f"Moved {filename} to {depth_path_base}")
        if filename.lower().startswith('camera_segmentation'):
            src_path = os.path.join(base_path, filename)
            dst_path_base = os.path.join(segmentation_path_base, filename)
            shutil.move(src_path, dst_path_base)
            print(f"Moved {filename} to {segmentation_path_base}")
        if filename.lower().startswith('bounding_box'):
            src_path = os.path.join(base_path, filename)
            dst_path_base = os.path.join(bounding_boxes_path_base, filename)
            shutil.move(src_path, dst_path_base)
            print(f"Moved {filename} to {bounding_boxes_path_base}")

def add_normals_and_colors_to_ply(input_path, output_path):
    # Load the original PLY file
    point_cloud = o3d.io.read_point_cloud(input_path)

    # Assuming point_cloud.points is of type o3d.utility.Vector3dVector
    num_points = len(point_cloud.points)
    
    # Generate random normals and colors
    normals = np.random.rand(num_points, 3) * 2 - 1  # Random normals between -1 and 1
    colors = np.random.rand(num_points, 3)  # Random colors between 0 and 1

    # Set normals and colors
    
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save the new PLY file
    o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=False, compressed=False)

    # Also create a version without normals if needed
    point_cloud.normals = o3d.utility.Vector3dVector([])  # Remove normals
    o3d.io.write_point_cloud(input_path.replace('global_points.ply', 'global_points_no_normals.ply'), point_cloud, write_ascii=False, compressed=False)

def create_cameras_txt(camera_info, output_path='cameras.txt', pinhole_width=1408, pinhole_height=1408, focal_length=610):
    if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            print(f"Created directory: {os.path.dirname(output_path)}")

    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# reading based on the ARIA description of distortion params: [k1, k2, k3, k4, k5 (ignore), p1, p2, s0, s1, s2, s3]\n")
        f.write("# writing based on COLMAPS description of distortion params for SIMPLE_PINHOLE : fx=fy cx cy \n")
        f.write("# writing based on COLMAPS description of distortion params for PINHOLE : fx fy cx cy \n")
        f.write("# writing based on COLMAPS description of distortion params for THIN_PRISM_FISHEYE : fx fy cx cy k1, k2, p1, p2, k3, k4, s0, s2 \n")
        # Extracting the parameters from the 'camera_info'
        fx = fy = focal_length  # Assuming fx and fy are the same
        cx = pinhole_width/2
        cy = pinhole_height/2
        # Distortion parameters order as per COLMAP's THIN_PRISM_FISHEYE
        # Following the array order [k1, k2, p1, p2, k3, k4, sx1, sy1]
        # Based on the ARIA description: [k1, k2, k3, k4, k5 (ignore), p1, p2, s0, s1, s2, s3]
        # Mapping: k1, k2, p1, p2, k3, k4, s0, s2
        k1, k2, k3, k4, k5, k6, p1, p2, s0, s1, s2, s3 = camera_info['distortion_params']
        # Create line for cameras.txt with the necessary parameters
        params = f"{fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3} {k4} {s0} {s2}"
        line = f"1 PINHOLE {camera_info['w']} {camera_info['h']} {fx} {fy} {cx} {cy}\n"
        f.write(line)
        line = f"# 2 THIN_PRISM_FISHEYE {camera_info['w']} {camera_info['h']} {params}\n"
        f.write(line)


def create_images_txt(frames, output_path='images.txt'):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        print(f"Created directory: {os.path.dirname(output_path)}")

    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, frame in enumerate(frames, start=1):
            # Extract transformation matrix and apply different transformations to rotation and translation
            transform = np.array(frame['transform_matrix'])
            rotation_matrix = transform[:3, :3]
            translation_vector = transform[:3, 3]

            # Apply 180 degree rotation about the x-axis to the rotation part
            #rotation_180_x = np.array([[1, 0, 0],
                                       #[0, -1, 0],
                                       #[0, 0, -1]])
            #rotation_matrix = rotation_matrix.dot(rotation_180_x)

            # Swap x and z in the translation vector
            # translation_vector = [translation_vector[2], translation_vector[1], translation_vector[0]]

            # Convert rotation matrix to quaternion (WXYZ order needed by COLMAP)
            quat = R.from_matrix(rotation_matrix).as_quat()  # XYZW
            quat = [quat[3], quat[0], quat[1], quat[2]]  # Convert to WXYZ

            # Write data to file
            f.write(f"{i} {' '.join(map(str, quat))} {translation_vector[0]} {translation_vector[1]} {translation_vector[2]} 1 {frame['file_path'].split('/')[-1]}\n\n")



def main():
    #put None to use aria transforms.json default
    pinhole_width,pinhole_height,focal_length = 1220,1220,None
    base_folder_path = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292'
    json_path = os.path.join(base_folder_path, 'transforms.json')
    transformed_output_json_path = os.path.join(base_folder_path, 'colmap_transforms.json')
    sample_vrs_file_path = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/raw_ADT_data/Apartment_release_meal_skeleton_seq137_M1292/video.vrs'
    images_txt_path = os.path.join(base_folder_path, 'colmap', 'sparse', '0', 'images.txt')
    cameras_txt_path = os.path.join(base_folder_path, 'colmap', 'sparse', '0', 'cameras.txt')
    aria_fisheye_images_path = os.path.join(base_folder_path)
    aria_pinhole_corrected_images_path = os.path.join(base_folder_path)
    input_ply_path = os.path.join(base_folder_path, 'global_points.ply')
    output_ply_path = os.path.join(base_folder_path, 'colmap', 'sparse', '0', 'points3D.ply')
    
    
    # Open the JSON file and load the data
    with open(json_path, 'r') as file:
        data = json.load(file)

    camera_info = data['frames'][0]  # Assuming all frames have the same camera info

    # Prepare camera instrinsics structure
    if pinhole_width is None:
        pinhole_width = data['frames'][0]['w']

    if pinhole_height is None:
        pinhole_height = data['frames'][0]['h']

    if focal_length is None:
        focal_length = data['frames'][0]['fl_x']

    process_aria_fisheye(aria_fisheye_images_path, aria_pinhole_corrected_images_path, sample_vrs_file_path,pinhole_width, pinhole_height, focal_length)

    # Start organizing, creating and copying folders and images
    organize_images_for_colmap(base_folder_path)
    # Create the cameras.txt in the specified output directory
    create_cameras_txt(camera_info, cameras_txt_path, pinhole_width, pinhole_height, focal_length)


    # Parameters dictionary (example values)
    camera_params = {
        'fx': camera_info['fl_x'], 'fy': camera_info['fl_y'],
        'cx': camera_info['cx'], 'cy': camera_info['cy'],
        'k1': camera_info['distortion_params'][0], 'k2': camera_info['distortion_params'][1], 'k3': camera_info['distortion_params'][2], 'k4': camera_info['distortion_params'][3], 'k5': camera_info['distortion_params'][4], 'k6': camera_info['distortion_params'][5],
        'p1': camera_info['distortion_params'][6], 'p2': camera_info['distortion_params'][7],
        's1': camera_info['distortion_params'][8], 's2': camera_info['distortion_params'][9], 's3': camera_info['distortion_params'][10], 's4': camera_info['distortion_params'][11]  # Assuming some values might be zero
    }

    # Create the images.txt in the specified output directory
    create_images_txt(data['frames'], images_txt_path)

    # Example usage
    convert_transforms(json_path, images_txt_path, transformed_output_json_path, input_ply_path, pinhole_width, pinhole_height, focal_length)

    # ply translation function call
    add_normals_and_colors_to_ply(input_ply_path, output_ply_path)

    

if __name__ == '__main__':
    main()

