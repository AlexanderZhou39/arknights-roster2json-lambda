import os
import torch

path_hubconfig = './yolov5'
path_weightfile = './models/weights.pt'

# initialize model into memory
model = torch.hub.load(
    path_hubconfig, 
    'custom',
    path=path_weightfile, 
    source='local'
)

def detect(img_bytes):
    results = model(img_bytes, size=1280)
    return results.pandas().xyxy[0]
