import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

def get_top_10_furthest_depth_values(folder_path):
    # Initialize a tensor to hold the top 10 values
    top_10_values = torch.full((10,), float('-inf'))  # Smallest possible values initially
    
    # Iterate through all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.png'):
            depth_image_path = os.path.join(folder_path, filename)
            
            # Load and process the depth image
            depth_image = Image.open(depth_image_path)
            depth_image = torch.from_numpy(np.array(depth_image).astype(np.float32))
            
            # Flatten the image and find the top 10 values in the current image
            current_top_10 = torch.topk(depth_image.flatten(), 10).values
            
            # Combine current top 10 with overall top 10 values
            combined_top_values = torch.cat((top_10_values, current_top_10))
            
            # Get the top 10 values from the combined tensor
            top_10_values = torch.topk(combined_top_values, 10).values

    return top_10_values

# Example usage
folder_path = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/ADT_depth'
top_10_furthest_values = get_top_10_furthest_depth_values(folder_path)
print("Top 10 furthest depth values:", top_10_furthest_values)
