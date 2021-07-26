# https://cloud.google.com/ai-platform/training/docs/using-containers
# If your training configuration uses NVIDIA A100 GPUs, then your container must use CUDA 11 or later

FROM nvidia/cuda:11.0.3-cudnn8-runtime
CMD nvidia-smi