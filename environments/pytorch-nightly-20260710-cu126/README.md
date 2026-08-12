# PyTorch nightly 2026-07-10 CUDA 12.6 environment

This image pins the official CPython 3.11 Linux CUDA 12.6 wheel for the
2026-07-10 PyTorch nightly. The Dockerfile verifies the wheel SHA-256 and
asserts the packaged Torch version, Git revision, CUDA version, and Triton
importability.

The final stage uses the official Python 3.11 slim runtime. Existing frozen
CUDA Python dependencies are copied from the managed April cu126 image in a
separate stage, with the previous Torch packages removed before installation.
No unpinned PyPI dependency install is performed.
