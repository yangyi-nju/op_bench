# PyTorch nightly 2026-07-10 CPU environment

This image pins the official CPython 3.11 Linux CPU wheel for the 2026-07-10
PyTorch nightly. The Dockerfile verifies the wheel SHA-256 before installation
and asserts both the packaged Torch version and Git revision.

The final stage uses the official Python 3.11 slim runtime. Existing frozen
dependencies are copied from the repository's managed CPU image in a separate
stage, with the previous Torch packages removed before the new wheel is
installed. This avoids an unpinned PyPI dependency install.

It is used as the binary/dependency substrate for exact Base-tree Python
overlays from early July 2026. Evaluation containers continue to run with
network access denied.
