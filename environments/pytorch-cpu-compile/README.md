# PyTorch CPU compile environment

This image extends `op-bench/pytorch-cpu:torch2.6.0-py311` with a working
GCC/G++ toolchain and Python 3.11 development headers. Use it for CPU
`python_overlay` tasks whose tests invoke `torch.compile` or Inductor C++ code
generation.

Build on the remote host:

```bash
docker build \
  -t op-bench/pytorch-cpu-compile:torch2.6.0-py311 \
  environments/pytorch-cpu-compile
```
