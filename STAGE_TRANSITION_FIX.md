# Stage Transition Fix: Preventing Gaussian Explosion

## Problem
When transitioning from `dynamics_RGB` to `fine_coloring` stage **without checkpointing**, dynamic Gaussians "explode" (jump to extreme positions) on the first training iteration.

## Root Cause
Creating a new optimizer in `training_setup()` loses Adam's momentum state (`exp_avg`, `exp_avg_sq`). Without accumulated momentum to moderate gradients, the first backward pass can produce unstable updates.

## Solution Implemented

### Primary Fix: Optimizer State Preservation
Modified `scene/gaussian_model.py :: training_setup()` to:
1. Save optimizer state before creating new optimizer
2. Restore momentum buffers after creating new optimizer
3. Match parameters by object identity (same tensor objects are reused across stages)

```python
# Before creating new optimizer
old_optimizer_state = None
if hasattr(self, 'optimizer') and self.optimizer is not None:
    old_optimizer_state = self.optimizer.state_dict()

# After creating new optimizer
if old_optimizer_state is not None:
    # Restore state for matching parameters...
```

## Testing the Fix

### Test 1: Full Training from Scratch
```bash
bash execution_scripts/HOI4D/Video1/run_video1_hoi.sh
```

**What to check:**
- Monitor the console for: `✓ Restored optimizer momentum state for X parameters across stage transition to 'fine_coloring'`
- Compare first few frames of:
  - `output/video1_bcv009_fulltrain/dynamics_RGB_render/images/` (last frames)
  - `output/video1_bcv009_fulltrain/fine_coloring_train_render/images/` (first frames)
- They should look similar (no sudden explosions)

### Test 2: With Debugging
```bash
# Add debugging to train_dynamic_depth.py
# After line: gaussians = GaussianModel(dataset.sh_degree, hyper)
# Add: from debug_stage_transition import instrument_gaussian_model_for_debugging
#      gaussians = instrument_gaussian_model_for_debugging(gaussians)
```

This will print detailed optimizer state info before/after each stage transition.

## Alternative/Complementary Fixes

If optimizer state restoration doesn't fully solve the issue, consider:

### Option A: Gradient Clipping During First N Iterations of New Stage
Add to `train_dynamic_depth.py` in the training loop:

```python
# After loss.backward(), before optimizer.step()
if stage == "fine_coloring" and iteration < first_iter + 10:
    # Clip gradients for first 10 iterations of new stage
    torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
```

### Option B: Learning Rate Warmup for New Stages
Modify the scheduler in `training_setup()` to have a warmup:

```python
# In fine_coloring stage, use warmup for first few iterations
def warmup_scheduler(iteration):
    base_lr = scheduler_args(iteration)
    if iteration < 100:
        warmup_factor = iteration / 100.0
        return base_lr * warmup_factor
    return base_lr
```

### Option C: Gradual Transition
Instead of abruptly switching stages, gradually blend learning rates:

```python
# Reduce dynamic LR to near-zero in last 100 iters of dynamics_RGB
# Then start fine_coloring with very small LR and ramp up
```

## Expected Behavior

### Before Fix
- `dynamics_RGB` stage ends: PSNR ~20-21, dynamic objects clear
- `fine_coloring` iteration 1: Loss spikes to 10000+, PSNR drops to ~6
- Visual: Gaussians scattered, confetti-like explosion

### After Fix
- `dynamics_RGB` stage ends: PSNR ~20-21, dynamic objects clear  
- `fine_coloring` iteration 1: Loss ~0.02-0.03, PSNR ~22-26
- Visual: Smooth continuation, no explosions

## Monitoring

Watch for these console messages:
- ✅ `✓ Restored optimizer momentum state for N parameters` - Fix working
- ⚠️  `No optimizer state restored` - Expected for first stage only
- ⚠️  `Failed to restore optimizer state` - Fix failed, check error

## Rollback

If this fix causes issues, revert by:
```bash
git diff scene/gaussian_model.py  # Check changes
git checkout scene/gaussian_model.py  # Revert if needed
```

## Related Files
- `scene/gaussian_model.py`: Main fix location
- `debug_stage_transition.py`: Debugging instrumentation
- `train_dynamic_depth.py`: Training script (stages defined here)
