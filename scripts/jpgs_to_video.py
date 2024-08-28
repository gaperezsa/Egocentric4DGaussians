import cv2
import os
import glob
from tqdm import tqdm
import time

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
    frame_height, frame_width, _ = first_image.shape

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))  # 30 fps

    try:
        # Iterate over the sorted list of file paths
        for file_path in tqdm(sorted_files):
            # Read image unchanged
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

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
    folder_path = "/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/output/bounding_box_depth_filtered/static_fine_render/images"
    output_video_path = "/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/output/bounding_box_depth_filtered/continious_training/bounding_boxes_filtered_depth_60000_video_30fps.mp4"
    create_video_from_png(folder_path, output_video_path)