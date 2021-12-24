# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np
from typing import List

class ObjectDetectionModel:
    def __init__(self):
        self.model: str
        self.names: List[str]
        self.stride: int
        self.imageSize: int
        self.device: str
        self.half: bool

class ObjectDetection:

    __models = {}
    __modelPaths = {
        'main_objects': 'yolov5s.pt',
        'sub_objects': 'yolov5s.pt',
    }
    
    def __init__(self, modelName: str):
        self.modelName = modelName
        self.deepsort = None
        self.deleted_tracker_objects = []

        if modelName not in ObjectDetection.__modelPaths:
            raise Exception(f"Invalid model: {modelName}")

        if modelName not in ObjectDetection.__models:
            raise Exception(f"Load model: {modelName}")
        

    @staticmethod
    def loadModel(modelName: str, imageSize: int = 640, device: str = "cpu", half: bool = False, dnn: bool = False):
        
        if modelName not in ObjectDetection.__modelPaths:
            raise Exception(f"Invalid model: {modelName}")

        if modelName in ObjectDetection.__models:
            return False

        # Initialize
        device = select_device(device)
        half &= device != 'cpu'  # half precision only supported on CUDA

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(ObjectDetection.__modelPaths[modelName], device=device, dnn=dnn)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        imageSize = check_img_size(imageSize, s=stride)  # check image size

        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.half() if half else model.float()

        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imageSize).to(device).type_as(next(model.parameters())))  # warmup

        modelObj = ObjectDetectionModel()
        modelObj.model = model
        modelObj.names = names
        modelObj.stride = stride
        modelObj.imageSize = imageSize
        modelObj.device = device
        modelObj.half = half

        ObjectDetection.__models[modelName] = modelObj
    

    def loadObjectTracker(self):
        
        config_deepsort = "./libs/yolov5_deepsort_pytorch/deep_sort/configs/deep_sort.yaml"

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.MODEL_TYPE,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    def detect(self, frame):

        result = []
        self.deleted_tracker_objects = []

        modelObj = ObjectDetection.__models[self.modelName]
        model = modelObj.model
        names = modelObj.names
        stride = modelObj.stride
        imgsz = modelObj.imageSize
        device = modelObj.device
        half = modelObj.half
        path = "???"
        augment = False
        visualize = False
        conf_thres = 0.3
        iou_thres = 0.5
        classes = None
        agnostic_nms = False
        max_det = 1000


        # Padded resize
        img = letterbox(frame, imgsz, stride=stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
                
            im0 = frame

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                if self.deepsort is None:
                    for *xyxy, conf, cls in reversed(det):
                        result.append((names[int(cls)], xyxy, None))
                else:
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    deleted_outputs = self.deepsort.tracker.deleted_tracks
                    
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            result.append((names[c], bboxes, id))
                    
                    if len(deleted_outputs) > 0:
                        for output in deleted_outputs: 
                            self.deleted_tracker_objects.append((names[int(output.class_id)], output.track_id))
                    
            else:
                if self.deepsort is not None:
                    self.deepsort.increment_ages()

        return result


"""ObjectDetection.loadModel("main_objects")
nb = ObjectDetection("main_objects")
nb.loadObjectTracker()


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        nb.detect(frame)"""
