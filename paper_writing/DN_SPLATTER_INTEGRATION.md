# DN-Splatter Integration for Egocentric 4D Gaussians

## Overview
This document describes the integration of depth and normal regularization techniques from the paper "DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing" (https://arxiv.org/pdf/2403.17822) into the Egocentric4DGaussians training pipeline.

## Changes Made

### 1. Data Loading (`scene/dataset_readers.py`)
- **Added `normal_map` field to `CameraInfo` NamedTuple**
- **Normal map loading**: Automatically loads normals from `{source_path}/normals/` directory
  - Expected filename pattern: `camera_normal_{frame_id}.npy`
  - Format: NumPy array (H, W, 3) with values in range [-1, 1]
  - Converted to PyTorch tensor (3, H, W) during loading

### 2. Camera Class (`scene/cameras.py`)
- **Added `normal_map` attribute**: Stores pre-computed normal maps per camera
- **Added `_image_gradient` cache**: Stores computed image gradients
- **Added `get_image_gradient()` method**: Computes RGB gradient once and caches it for reuse across iterations
  - Uses Sobel filters to compute gradient magnitude
  - Cached to avoid redundant computation (gradients are static per camera)

### 3. DN-Splatter Utilities (`utils/dn_splatter_utils.py`)
New module implementing all DN-Splatter loss functions and utilities:

#### A. Gradient Computation (for edge-aware weighting)
- `compute_image_gradient(image)`: Computes gradient magnitude using Sobel filters
  - Used for g_rgb = exp(-∇I) weighting

#### B. Gaussian Normal Computation (Eq. 6-7)
- `compute_gaussian_normals(quaternions, scales, means3D, camera_center, flip_to_camera)`
  - Computes normals from Gaussian geometry
  - Normal direction = smallest scaling axis (disc-like assumption)
  - Builds rotation matrix R from quaternion (Eq. 7)
  - Computes: n_i = R · OneHot(argmin(s_1, s_2, s_3))
  - Optionally flips normals to face camera

#### C. Normal Rendering
- `render_normal_map_from_gaussians(gaussians, viewpoint_camera, pipe, bg_color, stage, cam_type)`
  - Renders a normal map by:
    1. Computing normals from Gaussian geometry (rotation + scale)
    2. Transforming normals to camera space
    3. Using normals as "override_color" in existing renderer
    4. Alpha-compositing like RGB: N_hat = Σ n_i · α_i · T_i (Eq. 9)

#### D. Loss Functions

**Gradient-Aware Depth Loss (Eq. 4)**
- `gradient_aware_depth_loss(pred_depth, gt_depth, rgb_image, image_gradient, mask)`
  - Applies edge-aware weighting: g_rgb = exp(-∇I)
  - Uses logarithmic penalty: log(1 + ||pred - gt||_1)
  - Reduces loss at edges (high gradients), enforces more on smooth regions
  - Accepts pre-computed gradients for efficiency

**Normal Regularization Loss (Eq. 10-11)**
- `normal_regularization_loss(pred_normals, gt_normals, rgb_image, image_gradient, mask, lambda_l1, lambda_tv, use_gradient_aware)`
  - **L1 term**: Compares rendered normals with monocular normal priors
  - **TV smoothness**: Encourages smooth normal predictions at neighboring pixels
  - **Optional gradient-aware weighting**: Same as depth loss, reduces loss at edges

**Scale Regularization Loss (Eq. 8)**
- `scale_regularization_loss(scales, lambda_scale)`
  - Encourages disc-like Gaussians: L_scale = Σ ||argmin(s_i)||_1
  - Minimizes the smallest scaling axis to force flat, surfel-like shapes

### 4. Standard Depth Loss (`utils/loss_utils.py`)
- **Added `compute_image_gradient()`**: Sobel-based gradient computation
- **Added `gradient_aware_depth_loss()`**: Edge-aware depth loss with caching support

### 5. Training Script (`train_dynamic_depth.py`)

#### A. New Hyperparameters (`arguments/__init__.py`)
Added to `ModelHiddenParams`:
```python
use_gradient_aware_depth = True      # Use gradient-aware depth loss
normal_loss_weight = 0.05             # Weight for normal regularization
normal_l1_weight = 1.0                # L1 term within normal loss
normal_tv_weight = 0.01               # TV smoothness term within normal loss
```

#### B. Training Loop Modifications

**Data Collection** (per iteration):
- Collect `gt_normal_maps` from cameras that have them
- Collect `image_gradients` using cached `viewpoint_cam.get_image_gradient()`

**Depth Loss** (all stages with depth supervision):
- Replaced standard L1 depth loss with `gradient_aware_depth_loss`
- Uses cached image gradients for efficiency
- Falls back to standard L1 if `use_gradient_aware_depth=False`
- Applied in stages: `background_depth`, `background_RGB`, `dynamics_RGB`, `fine_coloring`

**Normal Loss** (RGB stages only):
- Renders normal maps from Gaussian geometry for each view
- Compares with GT monocular normals using L1 + TV loss
- Optionally applies gradient-aware weighting
- Applied in stages: `background_RGB`, `dynamics_RGB`, `fine_coloring`

**Scale Loss** (when normals are used):
- Applies scale regularization to encourage disc-like Gaussians
- Only active when `normal_loss_weight > 0`

**Total Loss**:
```python
loss = Ll1 + depth_loss + dynamic_mask_loss + normal_loss + scale_loss
```

## Theory Summary

### Gradient-Aware Depth Loss (Section 4.1)
- **Problem**: Depth sensors produce non-smooth edges at object boundaries
- **Solution**: Reduce depth loss at high-gradient regions (edges) using g_rgb = exp(-∇I)
- **Formula**: L_depth = g_rgb · (1/|D_hat|) · Σ log(1 + ||D_hat - D||_1)
- **Benefit**: More robust to edge artifacts, focuses regularization on smooth regions

### Normal Cues from Gaussian Geometry (Section 4.2)
- **Assumption**: Gaussians become flat, disc-like during optimization
- **Normal Definition**: Smallest scaling axis approximates surface normal direction
- **Computation**: n_i = R · OneHot(argmin(s_1, s_2, s_3))
  - R: Rotation matrix from quaternion
  - OneHot: Unit vector at position of minimum scale
- **Rendering**: Alpha-composite like colors: N_hat = Σ n_i · α_i · T_i
- **Supervision**: L1 loss with monocular normal priors (Omnidata-style)
- **Smoothness**: TV regularization on normal maps
- **Benefit**: No additional learnable parameters; gradients flow back to scale/rotation

### Scale Regularization (Eq. 8)
- **Purpose**: Force Gaussians to become disc-like surfels
- **Formula**: L_scale = Σ ||argmin(s_i)||_1
- **Effect**: Minimizes smallest scaling axis, making Gaussians flatter

## Usage Example

```bash
python train_dynamic_depth.py \\
    --source_path /path/to/data/with_monst3r/Video1 \\
    --model_path /path/to/output \\
    --background_depth_iter 20000 \\
    --background_RGB_iter 14000 \\
    --dynamics_depth_iter 10000 \\
    --dynamics_RGB_iter 5000 \\
    --fine_iter 20000
```

The script will automatically:
1. Load normals from `{source_path}/normals/`
2. Cache image gradients per camera
3. Apply gradient-aware depth loss
4. Render and supervise normals from Gaussian geometry
5. Apply scale regularization

## Expected Data Structure

```
Video1/
├── colmap/
│   ├── images/
│   │   ├── camera_rgb_0000.jpg
│   │   ├── camera_rgb_0001.jpg
│   │   └── ...
│   └── sparse/
├── depth/
│   ├── camera_depth_0000.png
│   ├── camera_depth_0001.png
│   └── ...
├── normals/
│   ├── camera_normal_0000.npy  # (H, W, 3) float32 in [-1, 1]
│   ├── camera_normal_0001.npy
│   └── ...
└── dynamic_masks/
    ├── camera_dynamics_0000.npy
    ├── camera_dynamics_0001.npy
    └── ...
```

## Key Benefits

1. **Gradient-Aware Depth**: More robust to sensor noise at edges
2. **Geometric Normals**: Leverages 3D Gaussian geometry without extra parameters
3. **Multi-Modal Supervision**: Depth + normals provide complementary geometric cues
4. **Efficiency**: Image gradients cached per camera (not recomputed every iteration)
5. **Disc-like Surfels**: Scale regularization encourages better surface representations

## Implementation Notes

- Normal rendering uses the existing rasterizer by treating normals as RGB colors
- Normals are transformed to camera space before rendering
- All losses are computed in batched form for efficiency
- Gradient caching reduces ~300x redundant gradient computations per training run
- Compatible with existing dynamic Gaussian training stages

## Future Improvements

1. Implement native normal rendering in CUDA rasterizer for better performance
2. Experiment with different normal estimation networks (currently assumes Omnidata-style)
3. Add depth-normal consistency loss (compute normals from depth, compare with rendered)
4. Adaptive weighting schedules for normal_loss_weight during training stages
