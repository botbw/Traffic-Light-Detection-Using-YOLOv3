import onnxruntime

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from easydict import EasyDict as edict

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import os
import logging
import json
import numpy as np

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
    t1 = torch_utils.time_synchronized()
    pred = sess.run(['modelOutput'], {'modelInput': img})
    t2 = torch_utils.time_synchronized()


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
