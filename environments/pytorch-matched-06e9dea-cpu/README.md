# PyTorch 06e9dea matched CPU runtime

This image pins the retained official CPython 3.11 PyTorch 2.7.0 CPU wheel used
to probe `pytorch__144073__vector_norm_scalar_overflow`.

The image includes a C++ toolchain and Ninja because the frozen P2P/F2P path
uses `torch.compile` and Inductor. The task overlays only
`torch/_refs/linalg/__init__.py` from Base Commit
`06e9deabb623e004eb6024e703a976c5748d51e6`.

Build on the registered remote host:

```bash
docker build \
  --tag op-bench/pytorch-matched-06e9dea:torch2.7.0-cpu-py311 \
  environments/pytorch-matched-06e9dea-cpu
```
