import os
import torch
import argparse
from torch2trt import torch2trt


def build_trt_model(model, input_tensor, args):
    print('Building TensorRT model [',args.quant,'] (again)...')
    test_model = model.cuda()
    input_tensor = input_tensor.cuda()
    if args.quant == 'fp16':
        test_model = test_model.half()
        input_tensor = input_tensor.half()
        test_model = torch2trt(test_model, [input_tensor], max_batch_size=64, fp16_mode=True,
                               max_workspace_size=1 << 30)
    elif args.quant == 'fp32':
        test_model = torch2trt(test_model, [input_tensor], max_batch_size=64, fp16_mode=False,
                               max_workspace_size=1 << 30)
    elif args.quant == 'int8':
        test_model = test_model.half()
        input_tensor = torch.tensor(input_tensor,dtype=torch.int8)
        test_model = torch2trt(test_model, [input_tensor], max_batch_size=64, max_workspace_size=1 << 30,
                               int8_mode=True)
    
    if not args.output:
        print('Model built successfully. Saving to tensorrt_state_dicts/{}_{}_trt.pth'.format(args.model, args.quant))
        torch.save(test_model.state_dict(), 'tensorrt_state_dicts/{}_{}_trt.pth'.format(args.model, args.quant))
    else:
        print('Model built successfully. Saving to tensorrt_state_dicts/{}_{}_{}_trt.pth'.format(args.model, args.quant, args.output))
        torch.save(test_model.state_dict(), 'tensorrt_state_dicts/{}_{}_{}_trt.pth'.format(args.model, args.quant, args.output))

    return test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorRT Inference Evaluation Script')
    parser.add_argument('--model', type=str, default='ernet', help='Model to build (ernet | squeeze-ernet | squeeze-redconv )')
    parser.add_argument('--output', type=str, default=None, help='Suffix to append at the output file name')
    parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained PyTorch weights (.pt) file')
    parser.add_argument('--quant', type=str, default='fp16', metavar='N',help='quantization scheme to use (default: fp16)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise OSError('CUDA is not available. Cannot build TensorRT engine!')

    model = None
    input_image = None

    if not os.path.exists('tensorrt_state_dicts'):
        os.mkdir('tensorrt_state_dicts')

    if args.model == 'ernet':
        if not args.weights:
            args.weights = 'weights/ernet.pt'
        input_image = torch.randn(1, 3, 240, 240)

    elif args.model == 'squeeze-ernet':
        if not args.weights:
            args.weights = 'weights/Squeeze-ernet-92f1score.pt'
        input_image = torch.randn(1, 3, 140, 140)

    elif args.model == 'squeeze-redconv':
        if not args.weights:
            args.weights = 'weights/Squeeze-ernet-redconv92acc.pt'
        input_image = torch.randn(1, 3, 140, 140)

    model = torch.load(args.weights, map_location='cpu')
    build_trt_model(model, input_image, args)