# Real Time Victim Localization for Disaster Management
Detecting humans and vehicles in a disaster classified image is used to estimate the number of victims stuck in disaster affected area using object detection techniques.
This repository contains pre trained models for object detection on AIDER Detect using [You Only Look Once](https://arxiv.org/abs/1506.02640)
![Victim Localization](/code/victim_localization/yolov3/detection_results.PNG)

## AIDER Detect
We have created a new dataset for object detection on aerial disaster imagery – named as AIDER Detect. It contains 2 Classes – person and vehicle and about 1428 images containing 6900 vehicles and 2100 humans in total. 

![AIDER_detect](/code/victim_localization/yolov3/aider_detect.jpg)

## Pre-trained Models
We have trained the following models based on YOLOv3 and YOLOv4 on AIDER Detect with following hyperparameters:
- Optimizer - Adam
- Activation function - Leaky ReLU
- Loss Function - Focal Loss
- Epochs - 2000

| Model | Precision | Recall | mAP@0.5 | FPS - GTX 1660 Ti |
|-------|-----------|--------|---------|-------------------|
| YOLOv3 | 0.307	  | 0.656  | 0.48	   |31.62|
| Tiny YOLOv3| 0.377| 0.267  |	0.196  | 94|
| Tiny YOLOv4| 0.221|	0.666  |	0.441	 |79.2|

Weights of pretrained models can be found in /weights folder

## Evaluation
To evaluate the model use `test.py` script
```
python test.py --cfg cfg/{model-aider-416.cfg} --weights weights/{model-aider-416.weights} --data data/aider.data 
```

## Inference
To run inference use `detect.py` script
```
python detect.py --cfg cfg/{model-aider-416.cfg} --weights weights/{model-aider-416.weights} --source /path/to/test/images --output /path/to/output/folder 
```

To convert checkpoints to weights use `pth_to_weights.py` 
```
python pth_to_weights.py --cfg cfg/{model-aider-416.cfg} --pt checkpoints/{model_ckpt_10.pth}
```
## Code Organization
The implementation in this repository is organized as following levels:

- **cfg:** contains architecture configuration files
- **data:** contains training, test and validation data
- **output:** contains inference output (Detected) images
- **utils:** contains utilities for pre and post processing
- **weights:** contains weights of pre trained models
