import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Function to apply Jet colormap to a depth image
def apply_jet_colormap(depth_image):
    # Convert to numpy array (ensure it's in 16-bit if it's 16-bit depth image)
    depth_array = np.array(depth_image)

    # Apply the Jet colormap directly based on raw depth values
    # Convert depth values to 8-bit range for colormap (Jet colormap requires values in the range [0, 255])
    depth_8bit = ((depth_array)/8513 * 255).astype(np.uint8)

    # Apply Jet colormap
    colored_image = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    return colored_image

# Paths
depth_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_kevis_corrected_z_depth'  # Path to the folder containing depth images
output_video = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/metric_crafter_kevis_corrected_z_depth_video.mp4'  # Output video file

# Get sorted list of images in the folder
image_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])

# Get the frame size from the first image using PIL
first_image = Image.open(os.path.join(depth_folder, image_files[0])).rotate(270)
width, height = first_image.size

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = 30  # Frames per second
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Process each image, apply Jet colormap and write to video
for image_file in tqdm(image_files):
    image_path = os.path.join(depth_folder, image_file)

    # Read the depth image using PIL
    depth_image = Image.open(image_path).rotate(270)

    # Apply Jet colormap
    colored_image = apply_jet_colormap(depth_image)

    # Write the frame to the video
    video_writer.write(colored_image)

# Release the video writer object
video_writer.release()

print(f"Video saved as {output_video}")
