
from dataclasses import dataclass
import threading
from typing import Any, Dict, List, cast
from PIL import Image, ImageDraw
from pathlib import Path

from projectaria_tools.core import mps
from projectaria_tools.core.data_provider import VrsDataProvider, create_vrs_data_provider
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

import shutil

import numpy as np
import csv
import json


ARIA_CAMERA_MODEL = "FISHEYE624"

# The Aria coordinate system is different than the Blender/NerfStudio coordinate system.
# Blender / Nerfstudio: +Z = back, +Y = up, +X = right
# Surreal: +Z = forward, +Y = down, +X = right
T_ARIA_NERFSTUDIO = SE3.from_matrix(
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
)

@dataclass
class AriaCameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_params: np.ndarray
    width: int
    height: int
    t_device_camera: SE3


@dataclass
class AriaImageFrame:
    camera: AriaCameraCalibration
    file_path: str
    t_world_camera: SE3
    timestamp_ns: float

@dataclass
class TimedPoses:
    timestamps_ns: np.ndarray
    t_world_devices: List[SE3]

def convert_id_image_to_color(image: np.ndarray, instances: dict) -> Image:
    # Create an empty array for the colored image
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    unique_ids = np.unique(image)

    for unique_id in unique_ids:
        if str(unique_id) in instances:
            color = instances[str(unique_id)]['color']
            # Create a mask for the current unique ID
            mask = image == unique_id
            color_image[mask] = color

    return color_image

def get_camera_calibs(provider: VrsDataProvider) -> Dict[str, AriaCameraCalibration]:
    """Retrieve the per-camera factory calibration from within the VRS."""

    factory_calib = {}
    name = "camera-rgb"
    device_calib = provider.get_device_calibration()
    assert device_calib is not None, "Could not find device calibration"
    sensor_calib = device_calib.get_camera_calib(name)
    assert sensor_calib is not None, f"Could not find sensor calibration for {name}"

    width = sensor_calib.get_image_size()[0].item()
    height = sensor_calib.get_image_size()[1].item()
    intrinsics = sensor_calib.projection_params()

    factory_calib[name] = AriaCameraCalibration(
        fx=intrinsics[0],
        fy=intrinsics[0],
        cx=intrinsics[1],
        cy=intrinsics[2],
        distortion_params=intrinsics[3:15],
        width=width,
        height=height,
        t_device_camera=sensor_calib.get_transform_device_camera(),
    )

    return factory_calib

def read_trajectory_csv_to_dict(file_iterable_csv: str) -> TimedPoses:
    closed_loop_traj = mps.read_closed_loop_trajectory(file_iterable_csv)  # type: ignore
    
    timestamps_secs, poses = zip(
        *[(it.tracking_timestamp.total_seconds(), it.transform_world_device) for it in closed_loop_traj]
    )

    SEC_TO_NANOSEC = 1e9
    return TimedPoses(
        timestamps_ns=(np.array(timestamps_secs) * SEC_TO_NANOSEC).astype(int),
        t_world_devices=poses,
    )

def to_aria_bounding_box_segmentation_frame(
    provider: VrsDataProvider,
    name_to_camera: Dict[str, AriaCameraCalibration],
    t_world_devices: TimedPoses,
    output_dir: Path,
    segmentation_provider = None,
    boxes = None,
    instances = None,
    vis_type = "segmentation"
) -> AriaImageFrame:
    name = "camera-rgb"

    camera_calibration = name_to_camera[name]
    stream_id = provider.get_stream_id_from_label(name)
    assert stream_id is not None, f"Could not find stream {name}"
    
    #human_id = '6810803442292599'
    time_diffs = []
    if vis_type == "segmentation":
        for box in boxes:
            object_uid = box['object_uid']
            if instances[object_uid]['instance_type'] == 'human' and str(stream_id) == box['stream_id']:
                box_timestamp_ns = int(box['timestamp[ns]'])
                image, image_data = segmentation_provider.get_image_data_by_time_ns(StreamId("400-1"),int(box['timestamp[ns]']),TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
                capture_time_ns = image_data.capture_timestamp_ns
                time_diff_ns = abs(capture_time_ns - box_timestamp_ns)
                time_diffs.append(time_diff_ns)
                if time_diff_ns < 1e9:  # less than 1 second
                    image = convert_id_image_to_color(image.to_numpy_array(), instances)
                    image = Image.fromarray(image)
                    draw = ImageDraw.Draw(image)

                    x_min = int(box['x_min[pixel]'])
                    x_max = int(box['x_max[pixel]'])
                    y_min = int(box['y_min[pixel]'])
                    y_max = int(box['y_max[pixel]'])
                    
                    # Draw bounding box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                    file_path = f"{output_dir}/camera_bounding_box_{capture_time_ns}.jpg"
                    image.save(file_path)
    else:
        for box in boxes:
            object_uid = box['object_uid']
            if instances[object_uid]['instance_type'] == 'human' and str(stream_id) == box['stream_id']:
                box_timestamp_ns = int(box['timestamp[ns]'])
                image, image_data = provider.get_image_data_by_time_ns(stream_id,int(box['timestamp[ns]']),TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
                capture_time_ns = image_data.capture_timestamp_ns
                time_diff_ns = abs(capture_time_ns - box_timestamp_ns)
                time_diffs.append(time_diff_ns)
                if time_diff_ns < 1e9:  # less than 1 second
                    image = Image.fromarray(image.to_numpy_array())
                    draw = ImageDraw.Draw(image)

                    x_min = int(box['x_min[pixel]'])
                    x_max = int(box['x_max[pixel]'])
                    y_min = int(box['y_min[pixel]'])
                    y_max = int(box['y_max[pixel]'])
                    
                    # Draw bounding box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                    file_path = f"{output_dir}/camera_bounding_box_rgb_{capture_time_ns}.jpg"
                    image.save(file_path)

    
    
    #threading.Thread(target=lambda: img.save(file_path)).start()
    
    #if segmentation_provider is not None:
    #    segmentation_data = segmentation_provider.get_image_data_by_time_ns(StreamId("400-1"),capture_time_ns,TimeDomain.DEVICE_TIME,TimeQueryOptions.CLOSEST)
    #    if abs(segmentation_data[1].capture_timestamp_ns - capture_time_ns) < 1e9:
    #        segmentation_array = segmentation_data[0].to_numpy_array()
    #        segmentation_file_path = f"{output_dir}/camera_segmentation_{capture_time_ns}.npy"
    #        threading.Thread(target=lambda: np.save(segmentation_file_path,segmentation_array)).start()
        

    # Find the nearest neighbor pose with the closest timestamp to the capture time.
    nearest_pose_idx = np.searchsorted(t_world_devices.timestamps_ns, capture_time_ns)
    nearest_pose_idx = np.minimum(nearest_pose_idx, len(t_world_devices.timestamps_ns) - 1)
    assert nearest_pose_idx != -1, f"Could not find pose for {capture_time_ns}"
    t_world_device = t_world_devices.t_world_devices[nearest_pose_idx]

    # Compute the world to camera transform.
    t_world_camera = t_world_device @ camera_calibration.t_device_camera @ T_ARIA_NERFSTUDIO

    return AriaImageFrame(
        camera=camera_calibration,
        file_path=file_path,
        t_world_camera=t_world_camera,
        timestamp_ns=capture_time_ns,
    )


def main(vrs_file, skeleton_boxes, instances_json_path, mps_data_dir, output_dir, vis_type) -> None:
    """Generate a nerfstudio dataset from ProjectAria data (VRS) and MPS attachments."""
    # Create output directory if it doesn't exist.
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = create_vrs_data_provider(str(vrs_file.absolute()))
    assert provider is not None, "Cannot open file"

    try:
        segmentation_provider = create_vrs_data_provider( str(vrs_file.absolute()).replace(str(vrs_file.absolute()).split("/")[-1],"segmentations.vrs") )
        shutil.copy(str(vrs_file.absolute()).replace(str(vrs_file.absolute()).split("/")[-1],"instances.json"), output_dir)
        print("Segmentation provider declared")
    except:
        segmentation_provider = None
        print("no segmentation vrs provided next to video.vrs")


    boxes_file = open(skeleton_boxes.absolute())
    boxes = csv.DictReader(boxes_file, delimiter=',')

    instances_file = open(instances_json_path.absolute())
    instances = json.load(instances_file)
    for instance_id in instances:
        instance = instances[instance_id]
        instance['color'] = tuple(np.random.choice(range(256), size=3))
        instances.update({instance_id: instance})

    name_to_camera = get_camera_calibs(provider)

    print("Getting poses from closed loop trajectory CSV...")
    trajectory_csv = mps_data_dir / "closed_loop_trajectory.csv"
    t_world_devices = read_trajectory_csv_to_dict(str(trajectory_csv.absolute()))

    name = "camera-rgb"

    # create an AriaImageFrame for each image in the VRS.
    print("Creating Bounding box frames...")
    to_aria_bounding_box_segmentation_frame(provider, name_to_camera, t_world_devices, output_dir, segmentation_provider, boxes, instances, vis_type)

if __name__ == "__main__":

    vrs_file = Path("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/raw_ADT_data/Apartment_release_meal_skeleton_seq137_M1292/video.vrs")
    skeleton_boxes = Path("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/raw_ADT_data/Apartment_release_meal_skeleton_seq137_M1292/2d_bounding_box_with_skeleton.csv")
    instances_json_path = Path("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/raw_ADT_data/Apartment_release_meal_skeleton_seq137_M1292/instances.json")
    mps_data_dir = Path("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/raw_ADT_data/Apartment_release_meal_skeleton_seq137_M1292/mps/slam")
    output_dir = Path("/home/gperezsantamaria/gperezsantamaria_2/Egocentric4DGaussians/data/ADT/NewApartmentMealSeq137M1292/box_rgb_vis")
    vis_type = "rgb" #segmentation or rgb

    main(vrs_file, skeleton_boxes, instances_json_path, mps_data_dir, output_dir, vis_type)