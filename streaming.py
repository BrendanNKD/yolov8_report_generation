import torch
from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument('--webcam-resolution',default =[1280,720],nargs=2,type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    model = YOLO("runs/detect/yolov8x_ppe_css_100_epochs/weights/best.pt")

    while True:
        ret,frame = cap.read()
        if not ret:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           continue
        results = model(frame)[0]

        for r in results:
            print(r.probs)
            annotator = Annotator(frame)
        
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                #stream here to a streaming databse 
                print(model.names[int(c)])
            frame = annotator.result()  
            cv2.imshow('yolov8',frame)
        
        if(cv2.waitKey(30)==27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()