ARG ACCOUNT_ID

# Dockerfile for training models using JAX
FROM $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fortuna:cuda-11.8.0-cudnn8-devel-ubuntu22.04

# Install python3
RUN apt update && apt install -y python3-pip

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip --no-cache-dir install --upgrade pip setuptools_rust

# Install ML Packages built with CUDA11 support
RUN ln -s /usr/lib/cuda /usr/local/cuda-11.8
RUN pip --no-cache-dir install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip --no-cache-dir install "aws-fortuna[transformers]"
RUN pip --no-cache-dir install sagemaker-training
RUN pip --no-cache-dir install smdebug
RUN pip --no-cache-dir install Jinja2

# Setting some environment variables related to logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
