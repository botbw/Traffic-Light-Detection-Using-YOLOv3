import argparse

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from easydict import EasyDict as edict

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect():
    imgsz = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    weights, half = opt.weights, opt.half

    # Initialize device
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Get names and colors
    names = load_classes(opt.names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0 , 118, 255)]

    # Run inference
    t0 = time.time()
    # BGR
    im0 = cv2.imread('preview_images/test3.jpeg')
    # RGB
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    trans = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)

    sample = {'image': im0}


    img = trans(**(sample))['image'].to(device)
    
    print(img.shape)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()

    # to float
    if half:
        pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
    for bbox in pred:
      if bbox is not None:
          # Rescale boxes from imgsz to im0 size
          bbox[:, :4] = scale_coords(img.shape[2:], bbox[:, :4], im0.shape).round()

          # Write results
          for *xyxy, conf, cls in bbox:
            print(('%g ' * 5 + '\n') % (cls, *xyxy))   

    # Print time (inference + NMS)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
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
    opt.device=''
    opt.view_img=False
    opt.save_txt=False
    opt.classes=None
    opt.agnostic_nms=False
    opt.augment=False
    # opt
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
