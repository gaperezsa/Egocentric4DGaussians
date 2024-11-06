import os, sys
import torch
from tqdm import tqdm
from time import time
from gaussian_renderer import render_dynamic_compare
import matplotlib.pyplot as plt

def movement_by_rendering(model_path, name, views, gaussians, pipeline, background, cam_type):
    fig, ax = plt.subplots()
    print("point nums:",gaussians._xyz.shape[0])
    per_gaussians_dynamic_movement = torch.zeros(gaussians._xyz.shape[0])
    os.makedirs(os.path.join(model_path, name),exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            recent_movement = render_dynamic_compare(view, gaussians, pipeline, background,cam_type=cam_type).detach().cpu()
            per_gaussians_dynamic_movement = per_gaussians_dynamic_movement + recent_movement

        if idx % 500 == 0: 
            ax.plot(list(range(gaussians._xyz.shape[0])), per_gaussians_dynamic_movement)

            ax.set(xlabel='gaussian', ylabel='total movement (meters)',
                title='Per gaussian movement while in frame')
            ax.grid()

            fig.savefig(os.path.join(model_path, name, "in_frame_movement.png"))

            plt.cla()

            # Plot the histogram with log scale
            plt.hist(per_gaussians_dynamic_movement, bins=50)
            plt.yscale('log')  # Apply logarithmic scale to the y-axis
            plt.xlabel('Total Movement (meters)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Gaussian Movements (Log Scale)')

            # Save the histogram
            fig.savefig(os.path.join(model_path, name, "in_frame_movement_histogram.png"))

            # Clear the plot for the next one
            plt.cla()
    
    # Compute the 95th percentile value
    threshold = torch.quantile(per_gaussians_dynamic_movement, 0.95)
    
    # Clip values in the tensor to the 95th percentile
    per_gaussians_dynamic_movement = torch.clamp(per_gaussians_dynamic_movement, max=threshold)
    
    ax.plot(list(range(gaussians._xyz.shape[0])), per_gaussians_dynamic_movement)

    ax.set(xlabel='gaussian', ylabel='total movement (meters)',
        title='Per gaussian movement while in frame')
    ax.grid()

    fig.savefig(os.path.join(model_path, name, "in_frame_movement.png"))
    
    plt.cla()

    # Plot the histogram with log scale
    plt.hist(per_gaussians_dynamic_movement, bins=50)
    plt.yscale('log')  # Apply logarithmic scale to the y-axis
    plt.xlabel('Total Movement (meters)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Gaussian Movements (Log Scale)')

    # Save the histogram
    fig.savefig(os.path.join(model_path, name, "in_frame_movement_histogram.png"))

    # Clear the plot for the next one
    plt.cla()
    
    # Use plt to colorize gaussians by movement
    norm = plt.Normalize()
    colors = torch.tensor(plt.cm.jet(norm(per_gaussians_dynamic_movement))[:,:3]).type(torch.float).to(gaussians._xyz.device)

    manual_normalized = per_gaussians_dynamic_movement/per_gaussians_dynamic_movement.max()
    manual_normalized [ manual_normalized > 0.75] = 0
    opacities = manual_normalized.unsqueeze(1).type(torch.float).to(gaussians._xyz.device)

    torch.save(per_gaussians_dynamic_movement,os.path.join(model_path, name, "per_gaussians_dynamic_movement.pt"))
    return per_gaussians_dynamic_movement, colors, opacities