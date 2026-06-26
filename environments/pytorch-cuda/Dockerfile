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

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/cusparselt/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusolver/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}
