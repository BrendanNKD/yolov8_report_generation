import os
import torch
from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting


def main():
    image_path = 'images/test.jpg'
    model = YOLO("runs/detect/yolov8x_ppe_css_100_epochs/weights/best.pt")
        # Load the image
    img = cv2.imread(image_path)

    # Perform inference
    results = model(img,save=True,project='results',name='predicted')
    for r in results:    
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            #stream here to a streaming databse 
            print(model.names[int(c)]) 


if __name__ == "__main__":
    main()