import os
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def compute_depth_correction(intrinsics: np.ndarray, image_dims: tuple) -> np.ndarray:
    # Create a grid of coordinates adjusted by the principal point
    xs, ys = np.meshgrid(np.arange(image_dims[1]), np.arange(image_dims[0]))
    xs = xs.astype(np.float32) - intrinsics[0, 2]
    ys = ys.astype(np.float32) - intrinsics[1, 2]
    
    # zs is the focal length in x (assuming fx = fy here)
    zs = np.ones_like(xs) * intrinsics[0, 0]
    
    # Stack to create 3D vectors for each pixel
    vector = np.stack([xs, ys, zs], axis=2)
    
    # Compute the normalization length and the correction factor
    lengths = np.linalg.norm(vector, axis=2)
    factors = intrinsics[0, 0] / lengths
    
    return factors

def apply_depth_correction(per_ray_depth, correction_factors):
    # Apply correction factors to convert per-ray depth to z-depth
    return per_ray_depth * correction_factors

def batch_convert_depth_images(input_folder, correction_factors):
    images_list = []
    filenames = []
    
    # Load and convert all images
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".png"):
            per_ray_depth = np.array(Image.open(os.path.join(input_folder, filename)), dtype=np.float32)
            z_depth = apply_depth_correction(per_ray_depth, correction_factors)
            images_list.append(z_depth)
            filenames.append(filename)
    
    return images_list, filenames

def multithread_write(image_list, filenames, path):
    os.makedirs(path, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(
                lambda img, fname: Image.fromarray(img.astype(np.uint16)).save(os.path.join(path, fname)),
                img, fname
            )
            for img, fname in zip(image_list, filenames)
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()  # Raise exceptions if any occurred during image saving

def process_folder(input_folder, output_folder, intrinsics, image_dims):
    # Compute the depth correction factors once for the image dimensions
    correction_factors = compute_depth_correction(intrinsics, image_dims)
    
    # Convert depth images with the correction factors
    image_list, filenames = batch_convert_depth_images(input_folder, correction_factors)
    multithread_write(image_list, filenames, output_folder)

# Parameters based on COLMAP's PINHOLE model
intrinsics = np.array([
    [610.9410078676575, 0, 610.0],
    [0, 610.9410078676575, 610.0],
    [0, 0, 1]
])

# Image dimensions
image_dims = (1220, 1220)  # Height and width

# Example usage
input_folder = "/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth/"
output_folder = "/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_kevis_corrected_z_depth"
process_folder(input_folder, output_folder, intrinsics, image_dims)
