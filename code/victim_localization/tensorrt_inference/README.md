## TensorRT Inference for YOLOv3 and v4 on AIDER-DETECT

### (Complete README to be added soon)


### Setup | Building YOLO layer plugin
Go to the `plugins/` directory and `make` the yolo layer plugin to generate `libyolo_layer.so`:
```
cd plugins
make
```

### Building TRT engine

Everything related to building the TensorRT-accelerated engine is in the `yolo/` directory. 

``` 
cd yolo
```

#### Converting trained YOLO model weights to ONNX
To convert from yolo model `.weights` file, place the model's `.cfg` file under the `cfg/` directory and its trained `.weights` file into `weights/`.
E.g for YOLOv4-Tiny (aider) with image size 416x416, run:

``` 
python yolo_to_onnx.py --model yolov4-tiny-aider-416 
```

Alternatively, path to the `.weights` file can also be specified using the `--weights` flag:
``` 
python yolo_to_onnx.py --model yolov3-tiny-aider-416 --weights ../yolov3/weights/yolov3-tiny-aider-416.weights
```
This will generate a `<model-name>.onnx` file in the `onnx/` directory.

#### Converting from ONNX to TensorRT engine
To convert the `<model-name>.onnx` file into a TensorRT engine (`.trt`), use the `onnx_to_tensorrt.py` script.
E.g for YOLOv4-Tiny (aider) with image size 416x416, run:
```
python onnx_to_tensorrt.py --model yolov4-tiny-aider-416
```
Building an engine takes a while. You will see your GPU memory inside `watch -n 0.1 nvidia-smi` go up. Once its done, `<model-name>.trt` file will
be generated inside `engines/` folder.

### Running TensorRT inference
TensorRT inference is done by the `trt_yolo.py` script under `tensorrt_inference/`.
To run inference on a directory of images use the `-d` flag followed by the `-i <path/to/images>`:
```
python trt_yolo.py --model yolov4-tiny-aider-416 -d -i ../yolov3/data/custom/test/images/
```

To run inference on a video:
```
python trt_yolo.py --model yolov4-tiny-aider-416 --video <path/to/video>
```

To run inference on live (laptop) webcam:
```
python trt_yolo.py --model yolov4-tiny-aider-416 --webcam
```

To run inference on Jetson TX2 onboard camera:
```
python trt_yolo.py --model yolov4-tiny-aider-416 --onboard
```

### Code organization
#### (TO be added)
