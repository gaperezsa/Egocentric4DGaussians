import os
import json
import shutil

def create_subset(data_dir, start_frame, end_frame):
    # Paths
    output_dir = f"{data_dir}_from_{start_frame}_to_{end_frame}"
    images_src = os.path.join(data_dir, "images")
    sparse_src = os.path.join(data_dir, "colmap/sparse/0")
    adt_depth_src = os.path.join(data_dir, "ADT_depth")
    depth_src = os.path.join(data_dir, "depth")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create necessary subdirectories
    images_dst = os.path.join(output_dir, "images")
    images_dst_2 = os.path.join(output_dir, "colmap/images")
    sparse_dst = os.path.join(output_dir, "colmap/sparse/0")
    adt_depth_dst = os.path.join(output_dir, "ADT_depth")
    depth_dst = os.path.join(output_dir, "depth")
    
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(images_dst_2, exist_ok=True)
    os.makedirs(sparse_dst, exist_ok=True)
    os.makedirs(adt_depth_dst, exist_ok=True)
    os.makedirs(depth_dst, exist_ok=True)

    # Copy and filter images based on frame range
    filtered_images = []
    for img_name in sorted(os.listdir(images_src)):
        frame_id = int(img_name.split('_')[2].split('.')[0])
        if start_frame <= frame_id <= end_frame:
            filtered_images.append(img_name)
            shutil.copy(os.path.join(images_src, img_name), images_dst)
            shutil.copy(os.path.join(images_src, img_name), images_dst_2)

    # Copy and filter depth images in ADT_depth and depth folders
    for folder_src, folder_dst in [(adt_depth_src, adt_depth_dst), (depth_src, depth_dst)]:
        for img_name in sorted(os.listdir(folder_src)):
            frame_id = int(img_name.split('_')[2].split('.')[0])
            if start_frame <= frame_id <= end_frame:
                shutil.copy(os.path.join(folder_src, img_name), folder_dst)

    # Rewrite the transforms.json file
    with open(os.path.join(data_dir, 'transforms.json'), 'r') as f:
        transforms_data = json.load(f)
    
    filtered_frames = [frame for frame in transforms_data['frames'] 
                       if start_frame <= frame['timestamp'] <= end_frame]
    transforms_data['frames'] = filtered_frames
    
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
        json.dump(transforms_data, f, indent=4)

    # Process images.txt file in colmap/sparse/0
    with open(os.path.join(sparse_src, 'images.txt'), 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    image_counter = 1
    for i in range(0, len(lines)):
        img_line = lines[i]
        if img_line == '\n':
            if new_lines[-1] != '\n':
                new_lines.append(img_line)
            continue
        
        parts = img_line.split()
        if parts[0] == '#':
            new_lines.append(img_line)
            continue
        
        img_name = parts[-1]
        frame_id = int(img_name.split('_')[2].split('.')[0])
        
        if start_frame <= frame_id <= end_frame:
            parts[0] = str(image_counter)  # Adjust image ID
            new_img_line = ' '.join(parts)
            new_lines.append(new_img_line + '\n')
            image_counter += 1

    with open(os.path.join(sparse_dst, 'images.txt'), 'w') as f:
        f.writelines(new_lines)

    # Copy other files in sparse (cameras.txt and points3D.ply)
    shutil.copy(os.path.join(sparse_src, 'cameras.txt'), sparse_dst)
    shutil.copy(os.path.join(sparse_src, 'points3D.ply'), sparse_dst)

    print(f"Subset created in {output_dir}")

# Usage example
create_subset("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292", 270971045579125, 271017371495125)
