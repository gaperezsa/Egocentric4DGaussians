# Depth Error Blame Tracking System

## Overview

Instead of using gradient-based approximations, we now use **direct per-pixel Gaussian attribution** from the rasterizer to assign blame for depth errors. The rasterizer returns `gaussian_idx [H, W]` which tells us exactly which Gaussian was most responsible for each pixel's depth value.

## Blame Accumulation During Training

### What We Track

For each Gaussian, we maintain three statistics:

1. **`_depth_error_sum`**: Cumulative sum of all depth errors this Gaussian was blamed for
   - If blamed once with 6cm error and once with 200cm error: sum = 206cm
   - Captures total "wrongness" accumulated over time

2. **`_depth_error_count`**: How many times this Gaussian was blamed
   - Tracks frequency of being responsible for depth errors
   - Can distinguish between "occasionally very wrong" vs "constantly slightly wrong"

3. **`_depth_error_max`**: Worst single depth error ever attributed to this Gaussian
   - Tracks catastrophic failures
   - Useful for identifying Gaussians with extreme outliers

### Per-Iteration Blame Assignment

During each training iteration:

```python
# In train_dynamic_depth.py:
# 1. Render depth map
depth_pred = render_with_dynamic_gaussians_mask(...)  # Returns gaussian_idx [H,W]

# 2. Compare against ground truth
depth_error = |depth_pred - depth_gt|

# 3. Find bad pixels (error > threshold)
bad_pixel_mask = (depth_error > threshold)

# 4. For each bad pixel, blame the responsible Gaussian
blamed_gaussian_ids = gaussian_idx[bad_pixels]  # Direct attribution!
error_magnitudes = depth_error[bad_pixels]

# 5. Accumulate statistics
gaussians.update_depth_error_stats(blamed_gaussian_ids, error_magnitudes)
```

## Blame Score Calculation

At pruning time, we compute a unified "blame score" that combines all three statistics:

$$\text{blame\_score} = (\text{error\_sum} + \text{max\_error}) \times \sqrt{\text{count}}$$

**Why this formula?**

- **error_sum + max_error**: Captures both consistent errors and catastrophic failures
  - 6cm + 200cm max = 206cm total (sum) + 200cm (max) = 406 points
  - This Gaussian is clearly problematic

- **√count**: Weights by frequency, but sublinearly
  - Avoids penalizing Gaussians blamed once with high error vs blamed 100 times with small errors
  - √count grows slower than linear, so frequency matters but error magnitude matters more

**Example scenarios:**

| Gaussian | Error Sum | Max Error | Count | Blame Score | Interpretation |
|----------|-----------|-----------|-------|-------------|-----------------|
| A        | 100cm     | 50cm      | 10    | (100+50)×√10 = 474 | Consistently wrong |
| B        | 200cm     | 200cm     | 1     | (200+200)×√1 = 400 | One catastrophic failure |
| C        | 50cm      | 10cm      | 1     | (50+10)×√1 = 60 | Minor occasional error |
| D        | 0cm       | 0cm       | 0     | 0 | Never blamed (innocent) |

→ Gaussian A is the worst (highest blame score), should be pruned first

## Pruning Strategies

### Strategy 1: Top X% Most Blamed (Recommended)

```python
# Prune worst 10% of Gaussians by blame score
gaussians.prune(
    min_opacity=0.005,
    max_scale=...,
    max_screen_size=...,
    depth_blame_percent=0.10  # NEW: Prune top 10% most blamed
)
```

**Advantages:**
- Adaptive to actual performance (prunes worst offenders)
- Fair comparison (same threshold for all)
- Preserves good Gaussians

### Strategy 2: Ever-Blamed (Strict)

```python
# Alternative: prune ANY Gaussian that was blamed even once
depth_blame_percent = 1.0  # Prune 100% of blamed Gaussians

# But combine with other gates:
prune_mask = blame_mask | opacity_mask | size_mask
```

**Advantages:**
- Aggressively removes contributors to errors
- Simple logic

**Disadvantages:**
- May be too strict (prunes good Gaussians with one bad frame)
- Doesn't account for error magnitude

### Strategy 3: Adaptive Threshold

```python
# Prune only Gaussians with blame_score > threshold
blame_scores = gaussians.compute_depth_blame_score()
mask = blame_scores > some_threshold
```

## Data Flow Summary

```
┌─────────────────────┐
│   Render Depth      │ ← Returns gaussian_idx [H,W] per pixel
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Compare vs GT depth │ ← Compute error magnitude per pixel
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Find bad pixels     │ ← error > threshold
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│ Direct Blame: gaussian_idx[bad_pixels]  │ ← WHO is responsible
│              depth_error[bad_pixels]    │ ← HOW WRONG are they
└──────────┬──────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Accumulate in update_depth_error_stats()     │
│ - _depth_error_sum += error_magnitude        │
│ - _depth_error_count += 1                    │
│ - _depth_error_max = max(_depth_error_max)   │
└──────────┬───────────────────────────────────┘
           │
           ▼ (every pruning iteration)
┌──────────────────────────────────────────┐
│ Compute blame_score for each Gaussian     │
│ blame_score = (sum + max) × √count        │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Get top X% most blamed Gaussians          │
│ Sort by blame_score, take worst           │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Prune them (remove from model)            │
│ Continue training without these Gaussians │
└──────────────────────────────────────────┘
```

## Key Advantages Over Gradient-Based Approach

1. **Ground Truth**: `gaussian_idx` is the exact Gaussian responsible (max α*T), not an approximation
2. **No Gradient Overhead**: No need to backprop through blame signal
3. **Direct Accountability**: Each error is attributed to the exact responsible Gaussian
4. **Flexible Scoring**: Can easily adjust blame score formula without retraining
5. **Interpretability**: Can visualize which Gaussians are problematic

## Monitoring & Debugging

### Log Output During Training

```
[ITER 400] Blamed 342 unique Gaussians across 5120 bad pixels (avg 14.9 blames per Gaussian)
[PRUNE] Removing 1288 Gaussians with highest blame scores
  Top blame score: 4521.384644
  Min blame score (of top 10.0%): 847.203125
```

### Accessing Blame Data

```python
# Get blame scores for analysis
blame_scores = gaussians.compute_depth_blame_score()  # [N, 1]

# Find most blamed Gaussians
top_indices, top_scores = gaussians.get_most_blamed_gaussians(topk_percent=0.1)

# Reset for next pruning interval
gaussians.reset_depth_error_stats()
```

## Configuration

Add to hyperparameters in `arguments/__init__.py`:

```python
class OptimizationParams:
    # Depth blame-based pruning
    self.depth_blame_prune_percent = 0.10  # Prune worst 10% by blame score
    self.depth_error_threshold_cm = 10.0   # Only blame pixels with >10cm error
```

Then in training:

```python
gaussians.prune(
    min_opacity=0.005,
    max_scale=...,
    max_screen_size=...,
    depth_blame_percent=opt.depth_blame_prune_percent  # Use configuration
)
```
