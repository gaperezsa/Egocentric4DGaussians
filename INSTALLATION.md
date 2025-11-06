# Installation Guide

## Prerequisites

- **CUDA**: 12.0+ (tested with CUDA 12.0)
- **Python**: 3.9
- **GCC**: 11.x (required for submodule compilation)

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n Gaussians4D python=3.9
conda activate Gaussians4D
```

### 2. Install PyTorch

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install GCC-11 (Critical for Submodule Compilation)

The submodules require GCC-11 due to pybind11 compatibility issues with newer GCC versions.

```bash
# Ubuntu/Debian
sudo apt-get install -y g++-11

# Verify installation
g++-11 --version
```

### 5. Initialize and Build Submodules

```bash
# Initialize git submodules
git submodule update --init --recursive

# Build submodules with GCC-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install -e submodules/simple-knn
pip install -e submodules/depth-diff-gaussian-rasterization
```

## Common Issues

### Issue: `pybind11` compilation errors
**Solution**: Ensure you're using GCC-11 by setting the environment variables before building submodules.

### Issue: CUDA version mismatch warnings
**Solution**: Minor version mismatches (e.g., CUDA 12.0 vs 12.1) are usually safe and can be ignored.

### Issue: Hardcoded paths in scripts
**Solution**: The training scripts now use dynamic path resolution with `$REPO_ROOT`.

## Running Training

```bash
bash execution_scripts/HOI4D/Video1/run_video1_hoi.sh
```

---

**Note**: This installation was successfully tested on Ubuntu 24.04 with CUDA 12.0 and PyTorch 2.1.2.
