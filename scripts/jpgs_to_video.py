import cv2
import os
import glob
from tqdm import tqdm
import time
import numpy as np

import re

def create_video_from_png(folder_path, output_video_path):
    # Print folder name being processed
    print(f"Processing folder: {folder_path}")
    # Search for .png files in the folder
    png_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    # Sort the list of file paths
    sorted_files = sorted(png_files)

    # Get the first image to determine frame size
    first_image = cv2.imread(sorted_files[0], cv2.IMREAD_UNCHANGED)
    #first_image = np.expand_dims(first_image,-1)
    frame_height, frame_width, _ = first_image.shape

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 15, (frame_width, frame_height))  # 30 fps

    try:
        # Iterate over the sorted list of file paths
        for file_path in tqdm(sorted_files):
            # Read image unchanged
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            #image = np.expand_dims(image,-1)
            # Rotate the rendered image if aria correction is required
            #image = np.rot90(image, k=3, axes=(0, 1))  # Rotate 270 degrees (k=3)
            # Write image to video
            video_writer.write(image)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save and release the video writer
        video_writer.release()

        # Add a small delay before printing the completion message
        time.sleep(1)

        # Check if the output video file exists
        if os.path.exists(output_video_path):
            print("Video creation complete.")
        else:
            print("Video creation failed. Check the output path.")
            
def numerical_sort_key(s):
    """ Extract number from filename for sorting. """
    return int(re.findall(r'\d+', s)[0])

if __name__ == "__main__":
    

    # aria
    folder_path = "/home/gperezsantamaria/data/Egocentric4DGaussians/output/chamfer_debugging_short_4/dynamics_RGB_train_render/images"
    output_video_path = "/home/gperezsantamaria/data/Egocentric4DGaussians/output/chamfer_debugging_short_4/dynamics_RGB_train_render/images/stage4_train.mp4"
    create_video_from_png(folder_path, output_video_path)