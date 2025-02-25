import numpy as np
from PIL import Image
import os


depth_or_rgb_images_path = "/home/gperezsantamaria/data/Egocentric4DGaussians/data/HOI4D/Video2/depth/"
demo_masks_output_dir = "/home/gperezsantamaria/data/Egocentric4DGaussians/data/HOI4D/Video2/dynamic_masks/"

os.makedirs(demo_masks_output_dir, exist_ok=True)

depth_image = Image.open(depth_or_rgb_images_path+os.listdir(depth_or_rgb_images_path)[0])
depth_image = (np.array(depth_image).astype(np.float32))/1000 #mm to meters

dynamic_mask = np.zeros_like(depth_image).astype(bool)

height, width  = dynamic_mask.shape[0], dynamic_mask.shape[1]
#dynamic_mask[int(height/7):int(6*height/7),int(4*width/7):] = True
dynamic_mask[int(4*height/7):,int(width/7):int(6*width/7)] = True

for name in os.listdir(depth_or_rgb_images_path):
    name = name.replace("camera_depth","camera_dynamics")
    name = name.replace("camera_rgb","camera_dynamics")
    np.save(demo_masks_output_dir+name.split('.')[0], dynamic_mask)