#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(
        description="Convert binary .npy masks → white-on-black .mp4 video"
    )
    parser.add_argument(
        "--npy_folder", "-i", required=True,
        help="Folder containing your .npy mask files"
    )
    parser.add_argument(
        "--output_video", "-o", required=True,
        help="Path to the output .mp4 video"
    )
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Frames per second (default: 15)"
    )
    args = parser.parse_args()

    # gather and sort all .npy files
    files = sorted(f for f in os.listdir(args.npy_folder) if f.lower().endswith('.npy'))
    if not files:
        raise RuntimeError(f"No .npy files found in {args.npy_folder}")

    # load first mask to get frame size
    first_mask = np.load(os.path.join(args.npy_folder, files[0]))
    if first_mask.ndim != 2:
        raise RuntimeError(f"Expected 2D array in {files[0]}, got shape {first_mask.shape}")
    h, w = first_mask.shape

    # set up video writer (white mask on black; we use color to be safe)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (w, h), isColor=True)

    for fn in files:
        mask = np.load(os.path.join(args.npy_folder, fn))
        if mask.shape != (h, w):
            # if any mismatch, resize nearest-neighbor
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        # scale to 0–255 and convert to BGR
        gray = (mask * 255).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame)

    writer.release()
    print(f"✔ Video saved to {args.output_video}")

if __name__ == "__main__":
    main()
