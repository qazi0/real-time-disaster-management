 We present two improved version of EmergencyNet architecture based on IEEE 2020 Paper  [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/abstract/document/9050881)
 - Squeeze ErNET 
 - Squeeze ErNET RedConv
 
## Atrous Convolution Feature Fusion (ACFF) Block 
![ACFF Block](/code/disaster_detection/resources/ACFF.png)

## Squeeze ErNET 
Squeeze ErNET is an improved version of EmergencyNET 
### Topology 
![Squeeze ErNET Topology](code/disaster_detection/resources/sq_ernet_topology.png)

### Results
Squeeze ErNET's performance on Aerial Imagery Dataset for Emergency Response:

| Precision | Accuracy | F1 Score | FPS GTX 1660 Ti | FPS TX2|
|-----| ---------|----------|---------|------|
| Pytorch Fp32| 92.5%	| 95.5% | 876.74 | 160.72|
| TRT FP32 | 88.45%	| 92.51%| 2951.13| 416.46	|
| TRT FP16|90.06%	|90.62%| 3430.17 |547.46|


## Squeeze ErNET RedConv Topology 
Squeeze ErNET RedConv is a compressed version of Squeeze ErNET with Reduced Convolutional layers which reduce the number of convolutional filters after every ACFF block. 

### Topology
![Squeeze ErNET RedConv Topology](code/disaster_detection/resources/redconv_topology.png)

### Results
Squeeze ErNET RedConv's performance on Aerial Imagery Dataset for Emergency Response:

| Precision | Accuracy | F1 Score | FPS GTX 1660 Ti | FPS TX2|
|-----|-------------|----------|-------|--------|
| Pytorch Fp32| 93%| 96% |783.29|142.11	|
| TRT FP32 | 94.1%	|93.281%| 2988.54|497.23	|
| TRT FP16|93.79%	|94.375%| 3196.18|569.37|


