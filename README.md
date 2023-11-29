# yolov8_report_generation
This repository contains an Yolov8 object detection inference and training. Integrated GPT2 model for context generation based on object detected

## Getting Started

Prerequisites
Ensure that you have the following resources on your system and resources ready :

- Python (3.11+)
- CUDA GPU

Follow these steps to get the for training:

```bash
python create_yaml.py
# then
python train.py
```


## Dataset
Dataset structure should follow:
    .
    ├── datasets
    |   ├── dataset_domain
    │     ├── test
    │     ├── train
    └─────├── valid
   

## Inference
Follow these steps to get the for streaming inference:

```bash
python streaming.py
```