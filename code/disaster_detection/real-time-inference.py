
import imutils
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from torch2trt import TRTModule
from imutils.video import WebcamVideoStream
from dataloaders.aider import aider_transforms, squeeze_transforms


def run_inference(model, transforms, args):

    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

        if args.trt:
            model = TRTModule()
            model.load_state_dict(torch.load('tensorrt_state_dicts/{}_{}_trt.pth'.format(args.model,args.quant)))

    # created a *threaded* video stream, allow the camera sensor to warmup
    prev_frame_time = 0
    fps_list = []

    print("Running inference...")
    vs = WebcamVideoStream(src=0).start()
    # loop over some frames...this time using the threaded stream
    while True:
        frame = vs.read()
        # frame = imutils.resize(frame, width=1366, height=768)
        img = transforms(Image.fromarray(frame))
        if args.cuda:
            img = img.cuda()

        img = img.reshape((1, 3, 240, 240))  if args.model == 'ernet' else img.reshape((1, 3, 140, 140))

        output = model(img)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time

        print('FPS: {:.3f}'.format(fps), end='\r')
        fps_list.append(fps)
        
        # if fps_update_index % 15 == 0:
        #     # puting the FPS count on the frame
        #     old_fps = fps 

        cv2.putText(frame, 'FPS: {:.3f}'.format(fps), (7, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: {:.3f}'.format(sum(fps_list) / len(fps_list)))
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time inference testing')
    parser.add_argument('--model', type=str, default='ernet')
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

    model = torch.load(args.weights, map_location='cpu').eval()

    run_inference(model, transforms, args)
