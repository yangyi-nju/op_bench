# PyTorch ff89ebc matched CUDA runtime

This image pins the retained official CPython 3.11 PyTorch 2.4.0 CUDA 12.4
wheel used to probe `pytorch__129154__exp_decomp_numerics`.

The wheel URL and SHA-256 are embedded in the Dockerfile. The image is only a
runtime asset: the task still overlays the target file from Base Commit
`ff89ebc50a738c734496393dc25313cf197fd0b4`, and full Baseline/Gold Admission
remains mandatory.

Build on the registered remote host:

```bash
docker build \
  --tag op-bench/pytorch-matched-ff89ebc:torch2.4.0-cu124-py311 \
  environments/pytorch-matched-ff89ebc-cu124
```
