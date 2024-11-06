import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from tqdm import tqdm

def load_npz_file(path):
    """Load the disparity map stored in .npz file"""
    data = np.load(path)['depth']  # Assumes the data is stored under 'depth'
    return data

def resize_image(image, target_size):
    """Resize a disparity map to match the size of a metric depth map"""
    return np.array(Image.fromarray(image.squeeze()).resize(target_size, resample=Image.BILINEAR))

def random_pixels(disparity_map, depth_map, num_pixels=20):
    """Select random pixels from the maps"""
    h, w = depth_map.shape
    pixels = [(random.randint(0, h-1), random.randint(0, w-1)) for _ in range(num_pixels)]
    
    # Depending on what we are looking for
    return [(1/disparity_map[i, j], depth_map[i, j]) for i, j in pixels]
    return [(disparity_map[i, j], depth_map[i, j]) for i, j in pixels]

# Define the function to fit: a + b/x
def func(x, a, b):
    return b/(a*x)

def process_npz_and_depth_maps(npz_file_paths, depth_map_folder, metric_depth_crafter_output):
    depth_paths = sorted(os.listdir(depth_map_folder))
    for idx, npz_file in tqdm(enumerate(npz_file_paths)):
        # Load the disparity map
        disparity_maps = load_npz_file(npz_file)  # Shape: (300, w, h, 1)
        num_maps = disparity_maps.shape[0]
        
        # Process 10 random samples from this set of 300 maps
        start_idx = idx * 300  # Index of the first corresponding depth map

        pairs = []
        for i in range(30):
            # Randomly select 30 images
            img_index = random.randint(0, num_maps - 1)
            depth_map_index = start_idx + img_index  # Corresponding depth map index
            
            # Load the metric depth map
            depth_map_path = os.path.join(depth_map_folder, depth_paths[depth_map_index])
            depth_map = np.array(Image.open(depth_map_path).rotate(270))
            
            # Resize the disparity map to match the depth map size
            disparity_map = resize_image(disparity_maps[img_index], depth_map.shape)
            
            # Get 10 random pixels (disparity, metric_depth) pairs
            pairs += random_pixels(disparity_map, depth_map, num_pixels=20)

        disparity_vals, metric_depth_vals = zip(*pairs)
        
        # Plot disparity vs. metric depth scatter
        plt.scatter(disparity_vals, metric_depth_vals, label='Data points')

        # Use curve_fit to fit the custom function
       #disparity_vals_np = np.array(disparity_vals)
        #metric_depth_vals_np = np.array(metric_depth_vals)
        
        # Fit the 1/x-like function: b / ax
        #popt, pcov = curve_fit(func, disparity_vals_np, metric_depth_vals_np)
        #a, b = popt
        #print(f"Fitted function: a = {a}, b = {b}")
        
        # Create range of disparity values for plotting the fitted curve
        #x_vals = np.linspace(min(disparity_vals), max(disparity_vals), 100)
        #y_vals_fitted = func(x_vals, *popt)
        
        # Plot the fitted curve
        #plt.plot(x_vals, y_vals_fitted, color='purple', label=f'Fit: {b:.3f}/{a:.3f}x')
        #plt.xlabel('Disparity')
        #plt.ylabel('Metric Depth')
        #plt.title(f'Disparity vs Metric Depth for NPZ {idx + 1}')
        #plt.legend()
        #plt.savefig(f"/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/scatter_npz_{idx+1}.png")
        #plt.close()
        
        # Polynomial regression to fit disparity vs. metric depth (degree 2)
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(np.array(disparity_vals).reshape(-1, 1))
        y = np.array(metric_depth_vals)
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(X_poly, y)

        # Print polynomial coefficients
        print(f"Polynomial regression: Coefficients {poly_reg_model.coef_}, Intercept {poly_reg_model.intercept_}")

        # Create range of disparity values for plotting the polynomial
        x_vals = np.linspace(min(disparity_vals), max(disparity_vals), 100)
        x_vals_poly = poly.transform(x_vals.reshape(-1, 1))
        y_vals_poly = poly_reg_model.predict(x_vals_poly)

        # Plot the polynomial curve
        plt.plot(x_vals, y_vals_poly, color='red', label='Polynomial Fit')
        plt.xlabel('Disparity')
        plt.ylabel('Metric Depth')
        plt.title(f'Disparity vs Metric Depth for NPZ {idx + 1}')
        plt.legend()
        plt.savefig(f"/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/scatter_npz_{idx+1}.png")
        plt.close()

        # Apply the polynomial regression model to predict metric depth for the entire disparity map
        for i, map in enumerate(disparity_maps):
            disparity_map_resized = resize_image(map, depth_map.shape)
            disparity_map_resized_flat = disparity_map_resized.flatten().reshape(-1, 1)
            disparity_map_resized_poly = poly.transform(disparity_map_resized_flat)
            predicted_depth = np.clip(poly_reg_model.predict(disparity_map_resized_poly),10,None).reshape(disparity_map_resized.shape) #predict but do not allow preediction closer than 1 milimiters

            # Save predicted depth as a PIL Image of mode 'I' (32-bit integer pixels)
            predicted_depth_img = Image.fromarray(predicted_depth.astype(np.int32), mode='I').rotate(90)
            predicted_depth_img.save(os.path.join(metric_depth_crafter_output, depth_paths[start_idx + i]))

# Example usage
npz_file_paths = [
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/depth_crafter_out_0/rgb_video.npz',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/depth_crafter_out_300/rgb_video.npz',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/depth_crafter_out_600/rgb_video.npz',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/depth_crafter_out_900/rgb_video.npz',
    '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/depth_crafter_output_2/depth_crafter_out_1200/rgb_video.npz'
]
depth_map_folder = '/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/ADT_depth'
metric_depth_crafter_output = "/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292_from_270971045579125_to_271017371495125/metric_depth_crafter_output_3"

os.makedirs(metric_depth_crafter_output, exist_ok=True)


os.makedirs(metric_depth_crafter_output, exist_ok=True)

process_npz_and_depth_maps(npz_file_paths, depth_map_folder, metric_depth_crafter_output)
