import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Function to mask objects that are 1 meter (1000 mm) away or less
def apply_depth_mask(depth_image, threshold_mm=1000):
    depth_array = np.array(depth_image)  # Convert to numpy array (in millimeters)
    
    # Create masks for areas within 1 meter (1000 mm) or less
    mask_near = (depth_array <= threshold_mm).astype(np.uint8) * 255  # Mask for near objects
    mask_far = (depth_array > threshold_mm).astype(np.uint8) * 255    # Mask for far objects

    # Create color versions of the masks (e.g., red for near objects, blue for far objects)
    mask_near_colored = np.zeros((*depth_array.shape, 3), dtype=np.uint8)
    mask_near_colored[:, :, 0] = mask_near  # Red channel for near objects (1 meter or less)

    mask_far_colored = np.zeros((*depth_array.shape, 3), dtype=np.uint8)
    mask_far_colored[:, :, 2] = mask_far  # Blue channel for far objects (greater than 1 meter)

    return mask_near_colored, mask_far_colored

# Paths
depth_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth'  # Path to the folder containing depth images
rgb_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/images'  # Path to the folder containing corresponding RGB images
output_video = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/meter_masked_Unidepth_video.mp4'  # Output video file

# Get sorted list of images in the folders
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.jpg')])

# Ensure there are the same number of depth and RGB images
assert len(depth_files) == len(rgb_files), "Number of depth and RGB images must match!"

# Get the frame size from the first RGB image using PIL
first_rgb_image = Image.open(os.path.join(rgb_folder, rgb_files[0]))
width, height = first_rgb_image.size

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = 30  # Frames per second
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Process each depth and RGB image, apply masks, and write to video
for depth_file, rgb_file in tqdm(zip(depth_files, rgb_files), total=len(depth_files)):
    depth_path = os.path.join(depth_folder, depth_file)
    rgb_path = os.path.join(rgb_folder, rgb_file)

    # Read the depth image using PIL
    depth_image = Image.open(depth_path)
    
    # Read the corresponding RGB image using OpenCV
    rgb_image = cv2.imread(rgb_path)

    # Apply depth masks (1 meter or less is masked in red, more than 1 meter in blue)
    mask_near, mask_far = apply_depth_mask(depth_image)

    # Blend the RGB image with the semi-transparent masks
    # Here, 0.7 is the weight of the original image and 0.3 is the weight of the mask (adjust for transparency)
    blended_image = cv2.addWeighted(rgb_image, 0.7, mask_near, 0.3, 0)
    blended_image = cv2.addWeighted(blended_image, 0.7, mask_far, 0.3, 0)
    blended_image = cv2.rotate(blended_image, cv2.ROTATE_90_CLOCKWISE)

    # Write the frame to the video
    video_writer.write(blended_image)

# Release the video writer object
video_writer.release()

print(f"Video saved as {output_video}")
