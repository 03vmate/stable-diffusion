#!/bin/sh
pip install diffusers==0.3.0 transformers scipy ftfy nvidia-pyindex onnx onnxruntime onnx-simplifier
pip install nvidia-tensorrt==8.4.3.1
pip install torch-tensorrt==1.2.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.2.0
dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.3.1-ga-20220813_1-1_amd64.deb
apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.3.1-ga-20220813/c1c4ee19.pub
apt update
apt install tensorrt -y
