import torch
import time
import argparse
import os
from torch.utils.data import DataLoader
from dataloaders.aider import AIDER
from dataloaders.aider import aider_transforms, squeeze_transforms
from torch2trt import TRTModule
import tensorrt as trt
from pytorch_lightning.metrics import F1


def load_data(args):
    transformed_dataset = AIDER("dataloaders/aider_labels.csv", args.root_dir, transform=aider_transforms if args.model == 'ernet' else squeeze_transforms)
    total_count = 6432
    train_count = int(0.5 * total_count)
    valid_count = int((0.5 - args.test_pc / 100) * total_count)
    test_count = total_count - train_count - valid_count
    train_set, valid_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                                   (train_count, valid_count, test_count))
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('Loaded all data.')
    return test_loader


def evaluate_performance(test_model, input_tensor, args):
    test_model.eval()
    time_list = []
    num_hits = 0
    correct = 0
    f1_list = []

    if args.cuda:
        print('Initializing inference on CUDA Device: ', args.gpu)
        # cuDnn configurations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        test_model = test_model.cuda()

        if args.trt:
            test_model = TRTModule()
            if args.state is not None:
                test_model.load_state_dict(torch.load(args.state))
            else:
                test_model.load_state_dict(torch.load('tensorrt_state_dicts/{}_{}_trt.pth'.format(args.model, args.quant)))


        # Load data
        test_loader = load_data(args)

        print('Running inference...')
        for batch_idx, (data, target) in enumerate(test_loader):
            tic = time.time()
            torch.cuda.synchronize()
            data = data.cuda()
            target = target.cuda()

            # FP16 Inference
            if args.trt and args.quant == 'fp16':
                output = test_model(data.half())

            # FP32 Inference
            else:
                output = test_model(data)

            preds = output.data.max(1, keepdim=True)[1]
            time_list.append(time.time() - tic)
            f1_list.append((preds,target))
            correct += preds.eq(target.data.view_as(preds)).sum()

       
        # Calculate F1 score
        f_score = F1(num_classes=5)
        score = 0
        for pred in f1_list:
            fixed_preds = pred[0].view(1, -1)
            score += f_score(fixed_preds[0].cpu(), pred[1].cpu()) * 100
        
        score /= len(f1_list)

        # Calculate total time
        time_list = time_list[1:]

        # Print Statistics
        print("     + Done ", len(test_loader.dataset), " iterations inference !")
        print("     + Total time cost: {}s".format(sum(time_list)))
        print("     + Average time cost: {}s".format(sum(time_list) / len(test_loader.dataset)))
        print('     + Accuracy: %.2f%%' % (100. * correct.float() / len(test_loader.dataset)))

        if args.trt:
            print("     + TensorRT Frames Per Second ({} iterations): {:.2f} FPS".format(len(test_loader.dataset),1 / (sum(time_list) / len(test_loader.dataset))))

        else:
            print("     + Frames Per Second ({} iterations): {:.2f} FPS".format(len(test_loader.dataset),1 / (sum(time_list) / len(test_loader.dataset))))
        
        print('     + F1 SCORE : {:.5f}'.format(score))

        if args.trt:
            print('     + Output data type: ', output.dtype)

    else:
        print('Running inference on CPU')

        # Load data
        test_loader = load_data(args)

        correct = 0
        f1_list = []

        for batch_idx, (data, target) in enumerate(test_loader):
            tic = time.time()
            output = test_model(data)
            preds = output.data.max(1, keepdim=True)[1]
            time_list.append(time.time() - tic)
            f1_list.append((preds,target))
            correct += preds.eq(target.data.view_as(preds)).sum()

       
        # Calculate F1 score
        f_score = F1(num_classes=5)
        score = 0
        for pred in f1_list:
            fixed_preds = pred[0].view(1, -1)
            score += f_score(fixed_preds[0].cpu(), pred[1].cpu()) * 100
        
        score /= len(f1_list)

        # Calculate total time
        time_list = time_list[1:]

        # Print Statistics
        print("     + Done ", len(test_loader.dataset), " iterations inference !")
        print("     + Total time cost: {}s".format(sum(time_list)))
        print("     + Average time cost: {}s".format(sum(time_list) / len(test_loader.dataset)))
        print('     + Accuracy: %.2f%%' % (100. * correct.float() / len(test_loader.dataset)))
        print("     + CPU Frames Per Second ({} iterations): {:.2f} FPS".format(len(test_loader.dataset),1 / (sum(time_list) / len(test_loader.dataset))))
        print('     + F1 SCORE : {:.5f}'.format(score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TensorRT Inference Evaluation Script')
    parser.add_argument('--model', type=str, default='ernet')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--test-pc', type=int, default=30, metavar='N',
                        help='percentage of data to use for testing (default: 30%)')
    parser.add_argument('--quant', type=str, default='fp16', metavar='N',
                        help='quantization scheme to use (default: fp16)')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',

                        help='gpu device to use (default: 0)')
    parser.add_argument('--trt', action='store_true', default=False, help='Perform TensorRT inference')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA inference')
    parser.add_argument('--root-dir', type=str, default='AIDER',
                        help='path to the root dir of AIDER')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model = None
    input_image = None

    if args.model == 'ernet':
        if not args.weights:
            args.weights = 'weights/ernet.pt'
        input_image = torch.randn(1, 3, 240, 240)
        print('ErNet Speed testing [',args.quant,'] with AIDER test images...')

    elif args.model == 'squeeze-ernet':
        if not args.weights:
            args.weights = 'weights/Squeeze-ernet-92f1score.pt'
        input_image = torch.randn(1, 3, 140, 140)
        print('Squeeze_ErNet Speed [',args.quant,'] with AIDER test images...')

    elif args.model == 'squeeze-redconv':
        if not args.weights:
            args.weights = 'weights/Squeeze-ernet-redconv92acc.pt'
        input_image = torch.randn(1, 3, 140, 140)
        print('Squeeze ErNet Speed (RedConv) [',args.quant,'] with AIDER test images...')

    model = torch.load(args.weights, map_location='cpu')

    # To speed up inference by disabling gradients
    with torch.no_grad():
        evaluate_performance(model, input_image, args)
