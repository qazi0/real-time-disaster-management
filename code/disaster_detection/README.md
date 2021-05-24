# Real-time Object Detection On FPGAs For Disaster Response
Quantized DNN for Classification + Segmentation of UAV Aerial Imagery for Disaster Detection and Response

## Architecture
EmergencyNet architecture has been implemented in PyTorch, based on CVPRW 2019 Paper [Deep-Learning-Based Aerial Image Classification for Emergency Response Applications Using Unmanned Aerial Vehicles
](https://arxiv.org/abs/1906.08716) and its 2020 revision as [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/abstract/document/9050881)

### Atrous Convolution Feature Fusion (ACFF) Block Architecture
![ACFF Block](/resources/ACFF.png)

![ErNet](/resources/MODEL.png)
## Dataset
Aerial Imagery Dataset for Emergency Response (AIDER) was used. It can be downloaded from [Zenodo](https://zenodo.org/record/3888300) or [Google Drive.](https://drive.google.com/file/d/1EUQ8BiTRn-ePsOUoB2WAPMAI9egbvtY-/view?usp=sharing)
![AIDER](/resources/AIDER-sample.png)

## Training
To launch training, goto model and run:

```python train.py --root-dir absolute/path/to/AIDER/folder --batch-size 16```
