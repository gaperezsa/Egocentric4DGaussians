import numpy as np
from PIL import Image
import sys

def process_image(input_path, output_path):
    # Open the image and convert it to a numpy array
    img = Image.open(input_path)
    img_np = np.array(img)

    # Normalize the image by dividing by the max value
    img_np = img_np / img_np.max()

    breakpoint()
    # Permute the array to have shape (Channels, Width, Height)
    img_np = np.expand_dims(img_np, axis=0)  # Add channel dimension

    # Repeat the array 3 times across the channel axis
    img_np = np.repeat(img_np, 3, axis=0)

    # Transpose back to (Width, Height, Channels) for saving
    img_np = np.transpose(img_np, (1, 2, 0))

    # Convert back to image
    img_processed = Image.fromarray((img_np * 255).astype(np.uint8))

    # Save the new image
    img_processed.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_image(input_path, output_path)
