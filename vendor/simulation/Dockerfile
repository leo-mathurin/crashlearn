# Crash&Learn Grand Prix — Docker image
#
# Stage: base — CPU-only, headless.  Used by: sim, train, tensorboard, viz.
#
# NOTE: f1tenth's setup.py is NOT used — it pins gym==0.19.0 and numpy<=1.22.
# Only the engine modules (f110_gym/envs/) are imported; deps are installed here.

FROM python:3.11-slim AS base

WORKDIR /app

# Build tools for numba + Pillow native extensions; tkinter for optional TkAgg
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        python3-tk \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Core deps — EXACT pins. numba JIT-compiles the physics and the LiDAR ray casting,
# and it declares a hard upper bound on numpy: a floating "numpy>=2.0" will eventually
# pull a numpy that numba refuses to import, and the whole simulation stops booting.
RUN pip install --no-cache-dir \
        "numpy==2.4.6" \
        "numba==0.65.1" \
        "scipy==1.17.1" \
        "Pillow==12.2.0" \
        "pyyaml==6.0.3" \
        "imageio==2.37.3" \
        "imageio-ffmpeg" \
        "matplotlib==3.11.0" \
        "opencv-python-headless" \
        "tensorboard" \
        "pytest"

# torch — CPU-ONLY build, from PyTorch's cpu index.
# The default PyPI wheel drags in ~2.7 GB of CUDA runtime that nothing here uses:
# the simulation is pure CPU and the compose file exposes no GPU. Same version, same
# API, same ONNX exporter — only the CUDA payload is gone.
# Training on GPU is a deliberate opt-in: reinstall torch from the CUDA index yourself.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0+cpu"

# ONNX toolchain — EXACT pins on purpose.
# The tournament loads every submission with the onnxruntime below. Loose pins would
# resolve to different versions depending on the build date, and a model written by a
# newer onnx than the runtime fails at load with "Unsupported model IR version".
# Announced maximum opset for submissions: 17.
RUN pip install --no-cache-dir \
        "onnx==1.22.0" \
        "onnxscript==0.7.1" \
        "onnxruntime==1.29.0"

# NOTE: no training framework is installed on purpose.
# The image ships the simulation and the ONNX toolchain — everything needed to run
# a race and to produce a model the tournament can load. Your training stack
# (gymnasium, stable-baselines3, your own loop, ...) is yours to install.

COPY . /app

# Engine modules are imported by path; PYTHONPATH points at the gym/ subdirectory
ENV PYTHONPATH=/app/gym
# Default headless matplotlib backend — can be overridden at runtime for TkAgg
ENV MPLBACKEND=Agg

