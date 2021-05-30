# Real Time Victim Localization for Disaster Managment
This repository provides the pretrained and optimized YOLO models for detecting disaster victims in aerial imagery.

![example](/code/victim_localization/yolov3/resources/detection_results.PNG)

## Features
Implementation in this directory has following features
- New object detection dataset for emergency response
- Pretrained YOLO models on AIDER Detect
- Quantized YOLO models

## Setup Requirement
The implementation of this repository has been tested on Ubuntu 18.04.5 LTS and Ubuntu 16.04.6 LTS with 5.4.0-73-generic and 4.4.0-210-generic kernels respectively. 

## Dataset
We have introduced [Object Detection Dataset for Emergency Response - ODDER](https://www.kaggle.com/maryamsana/yolov5emergencyresponse) with two classes.
- Person
- Vehicle

ODDER contains total 3930 labelled images containing 6900 vehicles and 2100 humans in total.

![ODDER](/code/victim_localization/yolov3/resources/aider_detect.jpg)


ODDER can be downloaded from kaggle. 
Before downloading dataset from kaggle follow the following steps:
```
pip install kaggle
cd ~/.kaggle
```
Go to your kaggle.com/{username}/account and click on generate new API token to download kaggle.json

```
mv Downloads/kaggle.json /.kaggle/kaggle.json
cd ~/.kaggle
chmod 600 kaggle.json
```
Now your kaggle authentication is complete.
Use the following command to download dataset for YOLOv3 and YOLOv4
```
kaggle datasets download -d kagglerx1/aiderdetectionyolo
```
Use the following command to download dataset for YOLOv5
```
kaggle datasets download -d maryamsana/yolov5emergencyresponse
```

# Code Organization
The implementation contains code on following levels:
- **tensorrt_inference:** contains tensorrt accelerated engines of pretrained YOLO models
- **yolov3:** contains pretrained YOLOv3 and YOLO4 models
- **yolov5:** contains pretrained YOLOv5 models

