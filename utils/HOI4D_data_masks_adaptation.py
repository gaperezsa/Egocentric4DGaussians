#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import cv2

def mask_to_binary(image_path):
    """
    Load a color PNG mask and return a 2D uint8 array:
      - 1 where any channel is non-zero (object)
      - 0 where all channels are zero (background)
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return np.any(arr != 0, axis=-1).astype(np.uint8)

def get_reference_shape(ref_path):
    """
    Determine the target (height, width) from:
      - a single image file, or
      - the first image in a folder
    """
    if os.path.isfile(ref_path):
        img = Image.open(ref_path)
    else:
        imgs = sorted([
            f for f in os.listdir(ref_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not imgs:
            raise RuntimeError(f"No images found in reference folder {ref_path}")
        img = Image.open(os.path.join(ref_path, imgs[0]))
    w, h = img.size
    return h, w

def main():
    parser = argparse.ArgumentParser(
        description="Convert color PNG masks → binary .npy, "
                    "resized to match a reference image."
    )
    parser.add_argument(
        "--masks_folder", "-i", required=True,
        help="Folder containing your PNG segmentation masks"
    )
    parser.add_argument(
        "--target", "-t", required=True,
        help="Path to a single reference image or a folder of reference images"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Folder where the resized .npy masks will be saved"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Determine target shape once
    target_h, target_w = get_reference_shape(args.target)

    # Gather and sort all input masks
    masks = sorted([
        f for f in os.listdir(args.masks_folder)
        if f.lower().endswith('.png')
    ])
    if not masks:
        raise RuntimeError(f"No .png files found in {args.masks_folder}")

    for i, fn in enumerate(masks):
        in_path = os.path.join(args.masks_folder, fn)
        mask = mask_to_binary(in_path)

        # Resize to the single reference shape
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        out_fn = f"camera_dynamics_{i:05d}.npy"
        out_path = os.path.join(args.output, out_fn)
        np.save(out_path, resized)

        print(f"[{i:05d}] Saved {out_fn} ({target_w}×{target_h})")

if __name__ == "__main__":
    main()
