# PyTorch nightly 2026-07-07 CPU environment

This image pins the official CPython 3.11 Linux CPU wheel for the 2026-07-07
PyTorch nightly. The Dockerfile verifies the wheel SHA-256 and asserts the
packaged Torch version and Git revision.

The final stage uses the official Python 3.11 slim runtime. Existing frozen
dependencies are copied from a managed repository image in a separate stage,
with the previous Torch packages removed before installation. No unpinned
PyPI dependency install is performed.
