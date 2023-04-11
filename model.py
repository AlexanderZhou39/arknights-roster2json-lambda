import os
import torch
from settings import INPUT_WIDTH

path_hubconfig = './yolov5'
path_weightfile = './models/yolov5.pt'

# initialize model into memory
model = torch.hub.load(
    path_hubconfig, 
    'custom',
    path=path_weightfile, 
    source='local'
)

def detect(img_bytes):
    results = model(img_bytes, size=INPUT_WIDTH)
    results_df = results.pandas().xyxy[0]
    return results_df[results_df['confidence'] > 0.6]
