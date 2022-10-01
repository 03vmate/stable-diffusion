#!/bin/sh
python -m onnxsim unet_v1_4_fp16_pytorch.onnx unet_v1_4_fp16_pytorch_sim.onnx
/usr/src/tensorrt/bin/trtexec --onnx=unet_v1_4_fp16_pytorch_sim.onnx --saveEngine=unet_v1_4_fp16_pytorch_sim.trt --fp16
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.9/dist-packages/torch/lib/
ldconfig
/usr/local/lib/python3.9/dist-packages/torch_tensorrt/bin/torchtrtc unet_v1_4_fp16_pytorch_sim.trt unet_v1_4_fp16_pytorch_sim.ts --embed-engine --device-type=gpu
