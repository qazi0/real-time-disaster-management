# Real Time Victim Localization with YOLOv3&v4
Detecting humans and vehicles in a disaster classified image is used to estimate the number of victims stuck in disaster affected area using object detection techniques.
This repository contains pre trained models for object detection on AIDER Detect using [You Only Look Once](https://arxiv.org/abs/1506.02640)
![Victim Localization](/code/victim_localization/yolov3/resources/detection_results.PNG)

## Setup Requirements
Inorder to install all requirements for this implementation use this command
```
pip install -U -r requirements.txt
```

## Dataset
We have used [Object Detection Dataset for Emergency Response- ODDER](https://www.kaggle.com/maryamsana/yolov5emergencyresponse) for training YOLO models

## Pre-trained Models
We have trained the following models based on YOLOv3 and YOLOv4 on AIDER Detect with following hyperparameters:
- Optimizer - Adam
- Activation function - Leaky ReLU
- Loss Function - Focal Loss
- Epochs - 2000

| Model | Precision | Recall | mAP@0.5 | FPS - GTX 1660 Ti |
|-------|-----------|--------|---------|-------------------|
| YOLOv3 | 0.307	  | 0.656  | 0.48	   |31.62|
| Tiny YOLOv3| 0.34| 0.243 |	0.153  | 94|
| Tiny YOLOv4| 0.221|	0.666  |	0.441	 |79.2|

Weights of pretrained models can be found in weights/ folder

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
- **resources:** Resources for readme
- **weights:** contains weights of pre trained models

This implementation is upbuilt upon git@github.com:roboflow-ai/yolov3.git
