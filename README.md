# Yolov11-Easy-Use-Object-Detection-Segmentation-Classification

# YOLOv11: Enhanced Object Detection-Custom Dataset

## Table of Contents

- [Introduction](#introduction)
- [Key Improvements in YOLOv11](#key-improvements-in-yolov11)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Inference](#model-inference)
- [Training Custom Data](#training-custom-data)
- [Acknowledgments](#acknowledgments)
- [Citations](#citations)
- [License](#license)

---

## Introduction

This repository implements the latest version of the **YOLO (You Only Look Once)** object detection model, specifically **YOLOv11**, as introduced by the Ultralytics team. YOLOv11 brings a variety of enhancements and optimizations compared to its predecessors, making it one of the most efficient and accurate models available for object detection.

The goal of this repository is to allow users to quickly and easily use YOLOv11 for object detection tasks, train custom datasets, and perform model inference with ease. Whether you're a researcher, developer, or enthusiast, YOLOv11 offers a powerful tool for your computer vision tasks.

---

## Key Improvements in YOLOv11

Compared to previous versions of YOLO, YOLOv11 introduces several key improvements aimed at enhancing performance, accuracy, and ease of use:

1. **Backbone Optimization**: YOLOv11 uses an updated backbone network that provides better feature extraction with lower computational cost, leading to faster inference and training.

2. **Improved Head Design**: The detection head has been refined to improve localization and classification accuracy, particularly for smaller objects that were challenging in earlier versions.

3. **Enhanced Feature Pyramid Network (FPN)**: The FPN architecture in YOLOv11 has been improved to handle multi-scale object detection better, ensuring more accurate detections across varying object sizes.

4. **Dynamic Anchor Boxes**: Instead of using fixed anchor boxes, YOLOv11 dynamically adjusts anchors during training to better match object aspect ratios, which increases the model's performance in various scenarios.

5. **Transformer Augmentations**: YOLOv11 incorporates advanced transformer-based techniques for improving long-range dependencies, which boosts the detection of objects even in cluttered or complex environments.

6. **Optimized Training Procedures**: By introducing novel training strategies and optimizations, YOLOv11 achieves better results with fewer epochs, making it more time-efficient to train on custom datasets.

7. **Improved Post-Processing**: Post-processing techniques such as Non-Maximum Suppression (NMS) have been enhanced to reduce false positives and improve overall detection quality.

---

## Features

- **Real-time Object Detection**: YOLOv11 is capable of processing images at high frame rates, making it suitable for applications requiring real-time detection.
- **Multi-Scale Object Detection**: Improved FPN allows detecting objects of varying sizes within the same image.
- **Custom Dataset Training**: Easily fine-tune YOLOv11 for specific tasks using your custom datasets.
- **Pretrained Models**: Access pretrained weights for a variety of tasks, speeding up the development process.
- **API Integration**: YOLOv11 can be seamlessly integrated into Python applications through a simple API.
- **Transformer-Augmented Detection**: Benefit from the latest advances in vision transformers for more accurate object detection.

---

## Installation


```bash
pip install ultralytics

```
Model training
Once you've installed all dependencies and downloaded the pretrained weights, you can start training using YOLOv11.
```bash
from ultralytics import YOLO

import torch

model = YOLO('/home/af/Vision/OBB/runs/obb/train7/weights/best.pt')
# Assuming 'model' is your trained model

# model.load_state_dict(torch.load('yolo_nas_s.pt'))
# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='map.yaml', epochs=500, imgsz=1024,batch = 14)
```

Model Inference
Now start making predictions using YOLOv11.
```bash
from yolov11 import YOLO

# Load model
model = YOLO(weights="yolov11-weights.pth")

# Perform inference
results = model.detect('path/to/image.jpg')

# Display results
model.show(results)
```


## Training Custom Data
You can also train the YOLOv11 model on your own dataset. Make sure your data is prepared in the correct format (e.g., COCO, YOLO annotation formats).

```bash
python train.py --data path/to/data.yaml --epochs 100 --batch-size 16 --img-size 640
```
This command will start training the model using your custom dataset. The --data flag should point to the .yaml configuration file that specifies your dataset's location, classes, and other relevant parameters

## Acknowledgments
This project utilizes the YOLOv11 architecture provided by Ultralytics. We would like to express our gratitude to the Ultralytics team for their continuous work on developing advanced models for object detection. The YOLO series has been instrumental in pushing the boundaries of real-time object detection, and YOLOv11 continues that tradition with further improvements in accuracy, speed, and flexibility.

The original repository and official models can be found on Ultralytics' GitHub page: Ultralytics YOLOv11

If you find this implementation useful, please consider starring their repository and citing their work as described below.


## Citations
If you use this codebase or YOLOv11 in your work, please cite the original Ultralytics repository as follows:
```bash
@article{YOLOv11_Ultralytics,
  author    = {Ultralytics},
  title     = {YOLOv11: Real-Time Object Detection},
  journal   = {GitHub repository},
  year      = {2024},
  url       = {https://github.com/ultralytics/ultralytics}
}

```
