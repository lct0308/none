import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from lib.models.YOLOP import MCnet, YOLOP

model = MCnet(YOLOP)
#model.load_state_dict(torch.load("test_scene_road/24epoch.pkl"))
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/data1/pbw/YOLOP-main/runs/MyDataset/_2023-07-28-03-26/epoch-149.pth").items()})

model = model.cpu()
model.eval()

input_shape = (3, 720, 1280)
batch_size = 1
dummy_input = torch.randn(batch_size, *input_shape)
onnx_file = 'laneatt.onnx'
torch.onnx.export(model,
                    dummy_input,
                    onnx_file,
                    opset_version=11,
                    input_names=["x",],
                    output_names=["output",])

