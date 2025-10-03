import os
from PIL import Image

# Paths to your directories
depth_folders = [
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_metric_out_0',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_metric_out_300',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_metric_out_600',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_metric_out_900',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_metric_out_1200'
]

# Path to the directory with correct timestamps
correctly_named_depth_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/ADT_depth'

# Path to save the processed images
output_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/crafter_depth'

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of timestamps (assuming they are in the same order)
names = sorted(os.listdir(correctly_named_depth_folder))

# Initialize image counter
image_counter = 0

for folder in depth_folders:
    images = sorted(os.listdir(folder))
    
    for image_name in images:
        # Open the image
        image_path = os.path.join(folder, image_name)
        image = Image.open(image_path)
        
        # Rotate the image by 90 degrees
        rotated_image = image.rotate(90)

        # Get the corresponding timestamp
        new_name = f"{names[image_counter].split('.')[0]}.png"

        # Save the image with the new name
        output_path = os.path.join(output_folder, new_name)
        rotated_image.save(output_path)

        # Increment the counter
        image_counter += 1
