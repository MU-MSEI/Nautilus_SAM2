# Nautilus SAM2

This repository contains a Docker and Kubernetes setup for running Meta's Segment Anything Model 2 (SAM2) on the Nautilus cluster.

## Architecture

The setup is optimized for Kubernetes deployment with persistent storage:

- **Docker Image**: Lightweight image (~5-6GB) without checkpoints
- **Checkpoints**: Stored on PersistentVolumeClaim (PVC) and shared across pods
- **Data**: Mounted from PVC for input/output

## Directory Structure

```
/develop/
├── code/       # SAM2 source code and your scripts
├── data/       # Input data and checkpoints (from PVC)
├── results/    # Output results
└── build/      # Build artifacts
```

## Setup Instructions

### 1. Build and Push Docker Image

```bash
# Build the image
docker build -t kovaleskilab/sam2:v1 -f docker/dockerfile .

# Push to registry
docker push kovaleskilab/sam2:v1
```

### 2. Create PersistentVolumeClaim

```bash
kubectl apply -f kube/pvc.yml
```

### 3. Download Checkpoints (One-time)

Download SAM2 model checkpoints to your PVC:

```bash
kubectl apply -f kube/podCheckpointDownload.yml

# Monitor the download
kubectl logs -f marshall-sam2-checkpoint-download

# Once complete, delete the pod
kubectl delete pod marshall-sam2-checkpoint-download
```

This downloads all SAM2 checkpoints (~2-3GB) to `/develop/data/checkpoints` on your PVC.

### 4. Launch Main Pod

```bash
kubectl apply -f kube/pod.yml

# Access the pod
kubectl exec -it marshall-sam2 -- /bin/bash
```

## Usage

Once inside the pod:

```bash
# Navigate to SAM2 directory
cd /develop/code/sam2

# Checkpoints are automatically linked from PVC
ls checkpoints/

# Run your notebooks or scripts
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Docker Image Optimizations

The Docker image has been optimized for size:

1. **Runtime base image**: Uses `pytorch:2.3.1-cuda11.8-cudnn8-runtime` instead of `devel` (~2GB smaller)
2. **No checkpoints**: Checkpoints stored on PVC instead of in image (~3GB smaller)
3. **Minimal dependencies**: Only essential system packages installed
4. **Clean build**: Removed git history and caches

**Before**: ~12-15GB  
**After**: ~5-6GB

## Available Checkpoints

SAM2 provides multiple model sizes:

- `sam2_hiera_tiny.pt` - Fastest, lowest memory
- `sam2_hiera_small.pt` - Balanced
- `sam2_hiera_base_plus.pt` - Better accuracy
- `sam2_hiera_large.pt` - Best accuracy, highest memory

## Files

- `docker/dockerfile` - Optimized Docker image
- `kube/pvc.yml` - PersistentVolumeClaim definition
- `kube/podCheckpointDownload.yml` - One-time checkpoint download pod
- `kube/pod.yml` - Main SAM2 pod
- `notebooks/` - Jupyter notebooks with SAM2 examples

## Resources

- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [SAM2 Paper](https://ai.meta.com/sam2/)

