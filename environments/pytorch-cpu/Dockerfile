FROM intel/intel-extension-for-pytorch:2.6.0-pip-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

RUN python -m pip install --no-cache-dir expecttest hypothesis pytest
