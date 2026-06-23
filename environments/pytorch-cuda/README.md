# pytorch-cuda environment

GPU-enabled PyTorch environment for CUDA-related task evaluation.

- Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- PyTorch: 2.6.0
- CUDA: 12.4
- cuDNN: 9
- Python: 3.11

## Build

On the remote GPU host:

```bash
cd environments/pytorch-cuda
docker build -t op-bench/pytorch-cuda:torch2.6.0-cu124-py311 .
```

## Verify

```bash
docker run --rm --gpus all op-bench/pytorch-cuda:torch2.6.0-cu124-py311 \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA A10G  (or whatever GPU is available)
```

## Hardware Requirements

- NVIDIA GPU (Ampere or newer recommended)
- nvidia-container-toolkit installed on host
- CUDA driver ≥ 12.1 (driver ≥ 530.x)
- ≥ 16 GB GPU memory (24 GB recommended)
