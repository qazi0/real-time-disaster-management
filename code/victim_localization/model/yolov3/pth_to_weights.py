from models import Darknet, save_weights, convert
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="cfg/yolov4-tiny-aider.cfg", help="path to model definition file")
parser.add_argument("--pt", type=str, default="checkpoints/yolov3_tiny_ckpt_290.pth", help="path to weights file")

opt = parser.parse_args()
# # Initiate model
# device = torch.device('cpu')

# model = Darknet(opt.cfg).to(device)

# print('Loading PyTorch weights: ',opt.pt)
# print('Loading CFG: ',opt.cfg)
# if opt.pt.endswith('.pth'):
# 	model.load_state_dict(torch.load(opt.pt, map_location=torch.device('cpu')))
# else:
# 	model.load_state_dict(torch.load(opt.pt, map_location=torch.device('cpu'))['model'])


# save_weights(model, 'weights/{}.weights'.format(os.path.basename(opt.cfg)[:-4]), cutoff=-1)
convert(opt.cfg, opt.pt)
