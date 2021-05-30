"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=2,
        help='number of object categories [2]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-d', '--detect-images', action='store_true', default=False,
        help='Test images')
    parser.add_argument(
        '-i', '--images-path', type=str, default='test/images',
        help='Test images')
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect_images(trt_yolo, conf_th, vis, test_imgs, model=''):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    import torch
    import glob
    test_imgs = glob.glob(test_imgs+'/*.jpg')

    fps = 0.0
    tic = time.time()
    time_start = tic

    import os
    if not os.path.exists('detections-' + os.path.basename(model)[:-4]):
        os.mkdir('detections-' + os.path.basename(model)[:-4])

    if not os.path.exists('detection-results-' + os.path.basename(model)[:-4]):
        os.mkdir('detection-results-' + os.path.basename(model)[:-4])
    
    detections_dir = 'detections-' + os.path.basename(model)[:-4] +'/'
    detection_results_dir = 'detection-results-' + os.path.basename(model)[:-4] +'/'

    classes = ['person','vehicle']
    

    for test_img in test_imgs:
        img = cv2.imread(test_img)
        if img is None:
            break

        # print('Img shape: ',img.shape)
        # input()
        # boxes, confs, clss = trt_yolo.detect(img, conf_th)
        outputs = trt_yolo.detect(img, conf_th)
        boxes, confs, clss = outputs
        boxes2 = torch.Tensor(boxes)
        detection_file_name = detection_results_dir + os.path.basename(test_img)[:-3] + 'txt'
        print('Saved ',detection_file_name)
        with open(detection_file_name,'w') as detection_file:
            for i in range(len(boxes)):
                out_line = ''
                out_line += (classes[int(clss[i])] + ' ')
                out_line += (str(confs[i]) + ' ')
                # print('Boxes[',i,'] = ',boxes[i])
                for box_dim in boxes[i]:
                    out_line += (str(int(box_dim.item())) + ' ')
                # print('Out line= ', out_line)
                out_line += '\n'
                detection_file.write(out_line)

            

        # print('Output shape= ',outputs)
        # print('Boxes= ',boxes.shape,' confs= ',confs)
        # input()
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        output_img_name = detections_dir+os.path.basename(test_img)
        # print('Saving ',output_img_name)
        cv2.imwrite(output_img_name,img)
        
    total_time = time.time() - time_start
    print('Processing speed (incl. File I/O): {} FPS'.format(len(test_imgs)/total_time))


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/engines/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/engines/%s.trt) not found!' % args.model)

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    model = args.model

    if args.detect_images:
        loop_and_detect_images(trt_yolo, conf_th=0.4, vis=vis, test_imgs=args.images_path, model=model)
    else:
        cam = Camera(args)
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        open_window(
                WINDOW_NAME, 'Camera TensorRT YOLO',
                cam.img_width, cam.img_height)

        loop_and_detect(cam, trt_yolo, conf_th=0.4, vis=vis)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
