import argparse
from pathlib import Path

def read_tum_trajectory(file_path):
    """
    Reads a TUM trajectory file that is assumed to have lines in the format:
      timestamp tx ty tz qw qx qy qz
    (lines beginning with '#' are ignored)
    
    Returns a list of poses where each pose is a dictionary:
      {'timestamp': float, 't': [tx, ty, tz], 'q': [qw, qx, qy, qz]}
    """
    poses = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                print(f"Skipping line with unexpected number of values: {line}")
                continue
            # Parse values; note that we assume the quaternion is given in [qw, qx, qy, qz] order.
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qw, qx, qy, qz = map(float, parts[4:8])
            poses.append({
                'timestamp': timestamp,
                't': [tx, ty, tz],
                'q': [qw, qx, qy, qz]
            })
    return poses

def write_colmap_images(poses, out_file):
    """
    Writes an images.txt file in the COLMAP format. The expected format is:
    
      # Image list with two lines of data per image:
      # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      1 qw qx qy qz tx ty tz 1 frame_000001.jpg
      
    Here, we use a fixed CAMERA_ID of 1 and generate a filename based on the image ID.
    """
    out_path = Path(out_file)
    with out_path.open("w") as f:
        # Write header comments
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        # Process each pose
        for idx, pose in enumerate(poses, start=0):
            # Format the numbers in a reasonable floating-point format.
            qw, qx, qy, qz = pose['q']
            tx, ty, tz = pose['t']
            # For the CAMERA_ID we use 1.
            camera_id = 1
            # Generate a filename (you can adjust the naming scheme as needed)
            name = f"camera_rgb_{idx:05d}.jpg"
            # Construct the line: image id, quaternion (w, x, y, z), translation (x,y,z), camera id, and name
            # All separated by spaces.

            line = f"\n"
            f.write(line)

            line = f"{idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n"
            f.write(line)
    print(f"Saved COLMAP images.txt to {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a TUM trajectory file (timestamp, tx, ty, tz, qw, qx, qy, qz) into a COLMAP images.txt file."
    )
    parser.add_argument("tum_file", help="Path to the input TUM trajectory file")
    parser.add_argument("out_file", help="Path to the output images.txt file")
    args = parser.parse_args()
    
    poses = read_tum_trajectory(args.tum_file)
    if not poses:
        print("No poses were read. Check the input file.")
        return
    write_colmap_images(poses, args.out_file)

if __name__ == "__main__":
    main()
