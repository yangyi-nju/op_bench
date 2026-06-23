FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /workspace

# Install test dependencies used by PyTorch test suite
RUN python -m pip install --no-cache-dir \
    expecttest \
    hypothesis \
    pytest \
    pytest-xdist \
    numpy
