# Real Time Victim Localization with YOLOv5
Detecting humans and vehicles in a disaster classified image is used to estimate the number of victims stuck in disaster affected area using object detection techniques. This repository contains pre trained models for object detection on AIDER Detect using [You Only Look Once](https://arxiv.org/abs/1506.02640)

![yolov5_detection](/code/victim_localization/yolov5/runs/detect/exp/normal_image0054.jpg)

## Setup Requirements
Inorder to install all requirements for this implementation use this command
```
pip install -U -r requirements.txt
```

## Dataset
We have used [Object Detection Dataset for Emergency Response- ODDER for training YOLO models](https://www.kaggle.com/maryamsana/yolov5emergencyresponse)


## Pre-Trained Model
We have trained the following models based on YOLOv5 on AIDER Detect with following hyperparameters:
- Optimizer - Adam
- Activation Function - Leaky ReLU
- Loss Function - Focal loss
- Epochs - 300

| Model | Precision | Recall | mAP@0.5 | FPS GTX 1660 Ti|
| ------|-----------|--------|---------|----------------|
| YOLOv5s| 0.584    |  0.662 | 0.612   | 70.49          |

Weights of pretrained model can be found in /weights folder

## Evaluation
To evaluate the model use `test.py` script

```
python test.py --weights weights/{model.pt} --data data/aider.yaml
```
## Inference 
To run inference using pre trained model use `detect.py` script
```
python detect.py --weights weights/{model.pt} --source /path/to/test/images
```
## Training
To train models on your dataset use `train.py` script
```
python train.py --weights weights/{model.pt} --cfg cfg/{model.yaml} --data data/{your-data.data}
```

## Image Verification
You use `img_verify.py` to verify and identify any corrupt images in data
```
python img_verify.py path/to/image/directory
```

## Code Organization
The implementation in this repository is organized as following levels:

- **cfg:** contains architecture configuration files
- **data:** contain data configurations
- **dataset:** contains training, test and validation data
-**runs:** contains inference output (Detected) images
- **utils:** contains utilities for pre and post processing
- **weights:** contains weights of pre trained models

This implementation is upbuilt upon git@github.com:ultralytics/yolov5.git 
