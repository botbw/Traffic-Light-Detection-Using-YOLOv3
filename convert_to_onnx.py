


from easydict import EasyDict as edict

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import os
import logging
import json

import torch.onnx 

from onnx import checker

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, opt.img_size, opt.img_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "navigasion_traffic.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
        #  opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

def init():
    global model, opt, device
    
    opt = edict()
    opt.cfg='cfg/yolov3-spp-6cls.cfg'
    opt.names='data/traffic_light.names'
    opt.weights='weights/best_model_12.pt'
    opt.source='preview_images/'
    opt.output='outputs'
    opt.img_size=512
    opt.conf_thres=0.3
    opt.iou_thres=0.6
    opt.fourcc='mp4v'
    opt.half=False
    opt.device=torch_utils.select_device(device='cpu')
    opt.view_img=False
    opt.save_txt=False
    opt.classes=None
    opt.agnostic_nms=False
    opt.augment=False
    # opt
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file


    # initialisze model
    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half = opt.weights, opt.half

    # Initialize device
    device = opt.device

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    model.eval()


if __name__ == '__main__':
  init()
  Convert_ONNX()
  checker.check_model('navigasion_traffic.onnx', True)
