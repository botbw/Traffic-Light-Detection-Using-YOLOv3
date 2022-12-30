import onnxruntime

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from easydict import EasyDict as edict

import torch
import torchvision

import os
import logging
import json
import numpy as np
import time
import cv2

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = True  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords



def detect(im0):
    ret = []
    # Run inference
    t0 = time.time()
    trans = A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                # ToTensorV2(p=1.0),
            ], p=1.0)
    sample = {'image': im0}
    img = trans(**(sample))['image']
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img[None]

    # Inference
    t1 = time_synchronized()
    pred = sess.run(['modelOutput'], {'modelInput': img})
    t2 = time_synchronized()


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
            ret.append(('%g ' * 5) % (cls, *xyxy))

    # Print time (inference + NMS)
    logging.info('Done. (%.3fs)' % (time.time() - t0))
    return ret



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global sess, opt
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl"
    )
    sess = onnxruntime.InferenceSession(model_path)

    opt = edict()
    # opt.cfg='cfg/yolov3-spp-6cls.cfg'
    # opt.names='data/traffic_light.names'
    # opt.weights='weights/best_model_12.pt'
    # opt.source='preview_images/'
    # opt.output='outputs'
    # opt.img_size=512
    opt.conf_thres=0.3
    opt.iou_thres=0.6
    # opt.fourcc='mp4v'
    # opt.half=False
    # opt.device=''
    # opt.view_img=False
    # opt.save_txt=False
    # opt.classes=None
    opt.agnostic_nms=False
    # opt.augment=False

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    data_json = json.loads(raw_data)
    data_str = data_json['image']
    data_arr = np.fromstring(data_str, np.uint8)

    im0 = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB).astype(np.float32)

    ans = detect(im0)
    logging.info("Request processed")
    return ans


# if __name__ == '__main__':
#   input = torch.randn(1, 3, 512, 512)
#   print(input.shape)  
#   input_name = sess.get_inputs()[0].name
#   print("input name", input_name)
#   input_shape = sess.get_inputs()[0].shape
#   print("input shape", input_shape)
#   input_type = sess.get_inputs()[0].type
#   print("input type", input_type)
#   print()
#   output_name = sess.get_outputs()[0].name
#   print("output name", output_name)
#   output_shape = sess.get_outputs()[0].shape
#   print("output shape", output_shape)
#   output_type = sess.get_outputs()[0].type
#   print("output type", output_type)

#   x = np.random.random((1, 3, 512, 512))
#   x = x.astype(np.float32)
#   res = sess.run([output_name], {input_name: x})
#   print(res[0].shape)

#   detect(x)
