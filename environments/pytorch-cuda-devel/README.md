# pytorch-cuda-devel environment

CUDA development environment for `cuda_kernel_build` runtime tier. Used when a
task requires modifying C++/CUDA kernel source files in PyTorch and needs an
in-place rebuild (`python setup.py develop --no-deps`).

## What it provides

- CUDA 12.4 toolkit + cuDNN 9 + nvcc (devel base image)
- Python 3.11
- PyTorch 2.6.0 pre-installed (baseline runtime; agent rebuilds from source)
- Build toolchain: gcc, ninja, cmake, ccache
- ccache configured at `/workspace/.ccache` with 10 GB cap
- `TORCH_CUDA_ARCH_LIST` set to cover Ampere/Ada/Hopper

## Build

On the remote GPU host:

```bash
cd environments/pytorch-cuda-devel
docker build -t op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311 .
```

Image size: ~12 GB (devel base + toolchain + PyTorch).

## Verify

```bash
docker run --rm --gpus all op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311 \
  bash -c "nvcc --version && python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```

## inplace_build flow

1. Workspace is rsynced to remote (full PyTorch source tree)
2. Agent modifies `.cpp` / `.cu` / `.h` files via the action interface
3. Before each test, `source_loading.build_command` runs (default:
   `cd {workspace_dir} && python setup.py develop --no-deps`)
4. ccache makes second-onward builds fast (only changed files recompile)

Typical timing on A10G:
- First build: 30-60 min
- Incremental build (single .cu file change): 2-5 min

## Hardware Requirements

- NVIDIA GPU (Ampere or newer required for sm_80+)
- nvidia-container-toolkit on host
- ≥ 24 GB system memory (PyTorch C++ link is memory-heavy)
- ≥ 50 GB free disk (build cache + objects)
