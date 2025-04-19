# Real-time Object Detection For Disaster Management From Aerial Imagery
A deep learning approach for disaster detection and victim localization for real time and effective disaster management.
(Quantized DNN for Classification of UAV Aerial Imagery for Disaster Detection and Response)

This repository has extended the work of CVPRW 2019 Paper [Deep-Learning-Based Aerial Image Classification for Emergency Response Applications Using Unmanned Aerial Vehicles
](https://arxiv.org/abs/1906.08716) and its 2020 revision as [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/abstract/document/9050881) for efficient disaster management. 

## Features
We have added the following additional features to the work of previously mentioned papers:
- Two new architectures for AIDER with improved performance and efficiency
- Quantized versions of proposed architectures 
- Accelerated inference speed
- A new dataset AIDER-Detect for victim localization
- YOLOv3 and YOLOv4 for AIDER-Detect
- Quantized YOLO (v3 & v4) for AIDER-Detect
- Power efficient networks


----

# Disaster Detection

In [code/disaster_detection](/code/disaster_detection/), we propose a method for real time disaster detection and classification using state of the art deep learning techniques. We have used an emergency response dataset called [Aerial Imagery Dataset for Emergency Response - AIDER ](https://zenodo.org/record/3888300#.YK94M6gzY2w) for training our deep learning models.

## Aerial Imagery Dataset for Emergency Response
The dataset contains 5 disaster classes. floods, collapsed buidling, fire, normal and traffic accidents. The aerial images of these disaster classes were collected from multiple sources such as the internet (e.g. google images, bing images, youtube, news agencies web sites, etc.), other databases of general aerial images, and images collected using our own UAV platform. During the data collection process the various disaster events were captured with different resolutions and under various condition with regards to illumination and viewpoint.
 It can be downloaded from [Zenodo](https://zenodo.org/record/3888300) or [Google Drive.](https://drive.google.com/file/d/1EUQ8BiTRn-ePsOUoB2WAPMAI9egbvtY-/view?usp=sharing)
![AIDER](code/disaster_detection/resources/AIDER-sample.png)

## Architecture
 We present two improved version of EmergencyNet architecture based on IEEE 2020 Paper  [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/abstract/document/9050881)
 - Squeeze ErNET 
 - Squeeze ErNET RedConv
 
### Atrous Convolution Feature Fusion (ACFF) Block 
![ACFF Block](code/disaster_detection/resources/ACFF.png)

## Inference
To run real time inference on your webcam stream use  `real-time-inference.py`
```
python real-time-inference.py --model {model} --weights weights/{model.pt}
```
To run inference on image of your choice use `aider-predict.py`
```
python aider-predict.py --model {model} --weights weights/{model.pt} --image path/to/image
```

### TensorRT Inference
TensorRT takes the network definition, performs optimizations including platform-specific optimizations, layer optimizations and generates the inference engine. 
To optimize a model for TensorRT inference add `--trt flag` 

### Quantized Inference 
Reduced precision networks tends to process faster without a significant loss in accuracy. To accelerate the inference we offer three quantization schemes for inference on
TensorRT.
- fp32
- fp16
- int8

The default quantization scheme is fp16. 
Add `--quant quant-scheme` flag for quantized inference

## Training
For training models use `train.py` 
```
python train.py --model {model} --root path/to/data 
```
For training on Google colab add `--collab` flag 

## Evaluate
You can evaluate performance of Pytorch models and their TensorRT engines on basis of their accuracy, F1 score and fps. 
To evaluate the models you can use `evaluate-classification-metrics.py` script
```
python evaluate-classification-metrics.py --model {mode} --weights weights/{model.pt} --root-dir path/to/data 
```
Add `--trt` and `--state path/to/pth/file` flags for evaluating TensorRT engine's performance

## Power Consumption
Power usage of model on Nvidia Jetson TX2 can be calculated using `calculate-power-usage.py` script
```
python calculate-power-usage.py --model {model} --weights weights/{model.pt} --root-dir path/to/data 
```
power usage script will give power trace graphs which are further used to estimate energy per frame for each model.

## Code Organization
The implementation in this repository is organized as following levels:
- **dataloaders:** contains the dataloader and transformations for AIDER
- **model:** contains the implementation of two different architectures squeeze ernet and squeeze ernet reduced conv 
- **model summary:** contains the architecture summary of implementated models
- **onnx:** contains the onnx files for implemented models
- **power usage plots:** contains the power consumption traces for implemented models in different quantization schemes
- **resources:** contains readme resources 
- **tensorrt state dicts** contain state dicts for TensorRT models 
- **weights** contains pretrained weights for implemented models

------

# Real Time Victim Localization
The [victim localization](/code/victim_localization/) directory provides the pretrained and optimized YOLO models for detecting disaster victims in aerial imagery.

![example](/code/victim_localization/yolov3/resources/detection_results.PNG)

## Features
Implementation in this directory has following features
- New Object Detection Dataset for Emergency Response ([ODDER](https://www.kaggle.com/maryamsana/yolov5emergencyresponse))
- Pretrained YOLO models on ODDER
- Quantized YOLO models

## Setup Requirement
The implementation of this repository has been tested on x86_64 Ubuntu 18.04.5 LTS 5.4.0-73-generic and Ubuntu 16.04.6 LTS 4.4.0-210-generic, as well as NVIDIA Jetson TX2 (aarch64 NVIDIA JetPack 4.4.2). 

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
Go to your `kaggle.com/{username}/account` and click on generate new API token to download `kaggle.json`

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

