AUTH_TOKEN = ""

import onnx
import torch
from diffusers import UNet2DConditionModel
import sys

width = int(sys.argv[1])
height = int(sys.argv[2])

assert(width % 8 == 0, "Width should be divisible by 8")
assert(height % 8 == 0, "Height should be divisible by 8")


unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16", subfolder="unet", use_auth_token=AUTH_TOKEN)
unet.cuda()

with torch.inference_mode(), torch.autocast("cuda"):
    inputs = torch.randn(2, 4, height // 8, width // 8, dtype=torch.half, device='cuda'), torch.randn(1, dtype=torch.half, device='cuda'), torch.randn(2, 77, 768, dtype=torch.half, device='cuda')

    # Export the model
    torch.onnx.export(unet,               # model being run
                    inputs,                         # model input (or a tuple for multiple inputs)
                    "unet_v1_4_fp16_pytorch.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_0', 'input_1', 'input_2'],
                    output_names = ['output_0'])
