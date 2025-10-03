from argparse import ArgumentParser
import json
import cv2
import numpy as np
import os
import multiprocessing

def overlay_mask_on_image(image, mask, color_true, color_false, alpha=0.1):
    overlay = image.copy()
    overlay[mask] = color_true
    overlay[~mask] = color_false
    # Blend the overlay with the original image
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output

def mask_given_segmentation_path(f):
    dynamic_mask = np.load(args.base_data_path+"/segmentation/"+f)
    static_mask = np.array([[segmentation_instances[str(x)]['motion_type'] == 'static' for x in i] for i in dynamic_mask])
    
    img = cv2.imread(args.base_data_path+"/images/"+f.replace("segmentation","rgb").replace(".npy",".jpg"))

    # Overlay the mask on the image
    output_image = overlay_mask_on_image(img, static_mask, color_true, color_false, alpha=0.5)

    # Save the result
    cv2.imwrite(args.base_data_path+"/dynamic_static_overlay_vis/"+"dynamic_static_"+f.split("_")[-1].replace(".npy",".jpg"), output_image)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument('--base_data_path', type=str, default="/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292")
    #parser.add_argument('--image_id', type=int, default=270938584096587)
    parser.add_argument('--image_id', type=int, default=0)

    args = parser.parse_args()

    # segmentation json
    try:
        segmentation_file = open(args.base_data_path + "/instances.json")
        segmentation_instances = json.load(segmentation_file)
        segmentation_instances['0'] = {'instance_id': 0, 'instance_name': 'empty', 'prototype_name': 'empty', 'category': 'nothing', 'category_uid': 0, 'motion_type': 'static', 'instance_type': 'human', 'rigidity': 'deformable', 'rotational_symmetry': {'is_annotated': False}, 'canonical_pose': {'up_vector': [0, 1, 0], 'front_vector': [0, 0, 1]}}
    except:
        print("no segmentation instances file found")

    if args.image_id != 0:
        img = cv2.imread(args.base_data_path+"/images/camera_rgb_"+str(args.image_id)+".jpg")
        dynamic_mask = np.load(args.base_data_path+"/segmentation/camera_segmentation_"+str(args.image_id)+".npy")
        static_mask = np.array([[segmentation_instances[str(x)]['motion_type'] == 'static' for x in i] for i in dynamic_mask])
        
        # Define colors (BGR format)
        color_true = (0, 255, 0)  # Green for static
        color_false = (0, 0, 255) # Red for dynamic

        # Overlay the mask on the image
        output_image = overlay_mask_on_image(img, static_mask, color_true, color_false, alpha=0.5)

        # Save the result
        cv2.imwrite("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/dynamic_static_"+str(args.image_id)+".jpg", output_image)

        print("Overlay image saved successfully.")
    else:
        # Define colors (BGR format)
        color_true = (0, 255, 0)  # Green for static
        color_false = (0, 0, 255) # Red for dynamic

        if not os.path.exists(args.base_data_path+"/dynamic_static_overlay_vis/"):
            os.makedirs(args.base_data_path+"/dynamic_static_overlay_vis/")

        pool_obj = multiprocessing.Pool()
        ans = pool_obj.map(mask_given_segmentation_path,os.listdir(args.base_data_path+"/segmentation/"))
        pool_obj.close()

        #for f in os.listdir(args.base_data_path+"/segmentation/"):
        #    dynamic_mask = np.load(args.base_data_path+"/segmentation/"+f)
        #   static_mask = np.array([[segmentation_instances[str(x)]['motion_type'] == 'static' for x in i] for i in dynamic_mask])
        #    
        #    img = cv2.imread(args.base_data_path+"/images/"+f.replace("segmentation","rgb").replace(".npy",".jpg"))

            # Overlay the mask on the image
        #    output_image = overlay_mask_on_image(img, static_mask, color_true, color_false, alpha=0.5)

            # Save the result
        #    cv2.imwrite(args.base_data_path+"/dynamic_static_overlay_vis/"+"dynamic_static_"+f.split("_")[-1].replace(".npy",".jpg"), output_image)

    