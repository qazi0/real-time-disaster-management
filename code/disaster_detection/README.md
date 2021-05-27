# Real-time Object Detection For Disaster Management From Aerial Imagery

This repository proposes a method for real time disaster detection and classification using state of the art deep learning techniques. We have used a emergency response dataset referred as [Aerial Imagery Dataset for Emergency Response - AIDER ](https://zenodo.org/record/3888300#.YK94M6gzY2w) for training our deep learning models.

## Aerial Imagery Dataset for Emergency Response
 It can be downloaded from [Zenodo](https://zenodo.org/record/3888300) or [Google Drive.](https://drive.google.com/file/d/1EUQ8BiTRn-ePsOUoB2WAPMAI9egbvtY-/view?usp=sharing)
![AIDER](resources/AIDER-sample.png)

## Architecture
 We present two improved version of EmergencyNet architecture based on IEEE 2020 Paper  [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/abstract/document/9050881)
 - Squeeze ErNET 
 - Squeeze ErNET RedConv
 
### Atrous Convolution Feature Fusion (ACFF) Block 
![ACFF Block](resources/ACFF.png)

## Environment Setup

## Inference
To run real time inference on your webcam stream use  `real-time-inference.py`
```
python real-time-inference.py --model model --weights weights/model.pt
```
To run inference on image of your choice use `aider-predict.py`
```
python aider-predict.py --model model --weights weights/model.pt --image path/to/image
```

### TensorRT Inference
Add `--trt flag` for inference on tensorRT engine 

### Quantized Inference 
We offer three quantization schemes for inference on tensorRT
- fp32
- fp16
- int8

The default quantization scheme is fp16. 
Add `--quant quant-scheme` flag for quantized inference

## Training

## Evaluate

## Code Organization

