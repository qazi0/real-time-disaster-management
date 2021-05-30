
import imutils
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from torch2trt import TRTModule
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
from dataloaders.aider import aider_transforms, squeeze_transforms


def run_inference(model, transforms, args):

    classes =['collapsed building','fire','flooded areas','normal','traffic incident']

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

    input_shape = (1,3,240,240) if args.model == 'ernet' else (1,3,140,140)

    print("Running inference...")

    if args.video:
        vs = FileVideoStream(args.video).start()
    else:
        vs = WebcamVideoStream(src=0).start()

    # loop over some frames...this time using the threaded stream
    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=args.width, height=args.height)
        img = transforms(Image.fromarray(frame))
        if args.cuda:
            img = img.cuda()

        img = torch.reshape(img, input_shape)

        if args.quant and args.trt:
            output = model(img.half())
        else:
            output = model(img)

        predicted_class = output.data.max(1,keepdim=True)[1]

        print("Predicted class is " + classes[predicted_class],end=' ')
        out = output[0]
        confidence = int((out[predicted_class]*100)[0][0].item())

        print("with confidence level : " + str(confidence) +'%', end=' \t')
        text = classes[predicted_class]+' with confidence level '+str(confidence)+'%' 
        cv2.putText(frame,text,(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,255),1,cv2.LINE_AA)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time

        print('FPS: {:.3f}'.format(fps))
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
    parser.add_argument('--model', type=str, default='squeeze-redconv')
    parser.add_argument('--video', type=str, default=None, help='path to a video to run inference on it')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--trt', action='store_true', help='Do TensorRT inference too')
    parser.add_argument('--width', type=int, default=640,
                        help='Window width')
    parser.add_argument('--height', type=int, default=480,
                        help='Window height')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the trained pytorch weights (.pt) file')
    parser.add_argument('--quant', type=str, default='fp16', metavar='N',
                            help='quantization scheme to use (default: fp16)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.trt and args.no_cuda:
        raise ValueError('TRT inference cannot be done on CPU! Please remove --no-cuda argument.')

    print(args)

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

    with torch.no_grad():
        run_inference(model, transforms, args)
