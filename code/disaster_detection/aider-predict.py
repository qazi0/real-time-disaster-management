import os
import cv2
import copy
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn

from PIL import Image
from skimage import io

from dataloaders.aider import aider_transforms,squeeze_transforms
import matplotlib.pyplot as plt

def predict(inp_model, transform, args):
	model.eval()
	with torch.no_grad():
            image = io.imread(args.image)
            data = transform(Image.fromarray(image))

            shape = (1,3,240,240) if args.model == 'ernet' else (1,3,140,140)

            data = torch.reshape(data,shape) 
            if args.cuda:
                data = data.cuda()

            output = inp_model(data)
            predicted_class = output.data.max(1,keepdim=True)[1]
            classes =['collapsed building','fire','flooded areas','normal','traffic incident']
            print("Predicted class is " + classes[predicted_class],end=' ')
            x = output[0]
            confidence = x[predicted_class]*100
            confidence = int(confidence[0][0].item())
            print("with confidence level : " + str(confidence) +'%')
            text = classes[predicted_class]+' with confidence level '+str(confidence)+'%' 
            image = cv2.putText(image,text,(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,255),1,cv2.LINE_AA)

            if args.trt:
                from torch2trt import TRTModule
                tensorrt_model = TRTModule()
                tensorrt_model.load_state_dict(torch.load('tensorrt_state_dicts/{}_{}_trt.pth'.format(args.model,args.quant)))
                
                if args.quant == 'fp16':
                    trt_output = tensorrt_model(data.half())
                else:
                    trt_output = tensorrt_model(data)

                predicted_class_trt = trt_output.data.max(1,keepdim=True)[1]

                print("Predicted class (TRT) is " + classes[predicted_class_trt], end=' ')
                x = trt_output[0]
                confidence_trt = x[predicted_class_trt]*100
                confidence_trt = int(confidence_trt[0][0].item())
                print("with confidence level : " + str(confidence_trt) +'%')

                text_trt = classes[predicted_class_trt]+' with TRT confidence level '+str(confidence_trt)+'%'
                image = cv2.putText(image,text_trt,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,255),1,cv2.LINE_AA)
            
            plt.imshow(image)
            plt.show()

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Predict AIDER images')
    parser.add_argument('--model', type=str, default='ernet')
    parser.add_argument('--image', type=str, default='AIDER/fire/fire_image0003.jpg')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--trt', action='store_true', help='Do TensorRT inference too')

    parser.add_argument('--weights', type=str, default=None,
                        help='path to the trained pytorch weights (.pt) file')
    parser.add_argument('--quant', type=str, default='fp16', metavar='N',
                            help='quantization scheme to use (default: fp16)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.model == 'ernet':
        transforms = aider_transforms
        if args.weights is None:
            args.weights = 'weights/ernet.pt'

    elif args.model == 'squeeze-ernet':
        transforms = squeeze_transforms
        if args.weights is None:
            args.weights = 'weights/Squeeze-ernet-92f1score.pt'

    elif args.model == 'squeeze-redconv':
        transforms = squeeze_transforms
        if args.weights is None:
            args.weights = 'weights/Squeeze-ernet-redconv92acc.pt'

    model = torch.load(args.weights, map_location='cpu')

    if args.cuda:
        model = model.cuda()

    predict(model,transforms,args)


