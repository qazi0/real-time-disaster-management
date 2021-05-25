import os
import onnx
import torch
import argparse

def generate_and_verify_onnx(args):

    ONNX_FILE_PATH = 'onnx/{}'.format(args.model)
    if args.model == 'ernet':
        if args.weights is None:
            args.weights = 'weights/ernet-96f1scor.pt'

        input_tensor = torch.randn(args.batch_size, 3, 240, 240, requires_grad=True)

    elif args.model == 'squeeze-ernet':
        if args.weights is None:
            args.weights = 'weights/Squeeze-ernet-92f1score.pt'

        input_tensor = torch.randn(args.batch_size, 3, 140, 140, requires_grad=True)

    elif args.model == 'squeeze-redconv':
        if args.weights is None:
            args.weights = 'weights/Squeeze-ernet-redconv92acc.pt'

        input_tensor = torch.randn(args.batch_size, 3, 140, 140, requires_grad=True)

    model = torch.load(args.weights, map_location='cpu')
    model.eval()

    print('Generating ONNX file for {}...'.format(args.model))
    if not args.no_dynamic:
        ONNX_FILE_PATH += '-dynamic.onnx'
        torch.onnx.export(model, input_tensor, ONNX_FILE_PATH, input_names=['input_1'], output_names=['output_1'],
                      export_params=True,
                      opset_version=10, do_constant_folding=True,
                      dynamic_axes={'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}})
    else:
        ONNX_FILE_PATH += '.onnx'
        torch.onnx.export(model, input_tensor, ONNX_FILE_PATH, input_names=['input_1'], output_names=['output_1'],
                      export_params=True,
                      opset_version=10, do_constant_folding=True)

    print('Generated ONNX file ', ONNX_FILE_PATH, ' successfully!')

    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
    print('Verified ONNX file ', ONNX_FILE_PATH, ' successfully!')
    onnx.helper.printable_graph(onnx_model.graph)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='onnx model conversion')
    parser.add_argument('--model', type=str, default='ernet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the trained pytorch weights (.pt) file')
    parser.add_argument('--no-dynamic', action='store_true', help='disables dynamic axes')
    args = parser.parse_args()

    if not os.path.exists('onnx'):
        os.mkdir('onnx')

    generate_and_verify_onnx(args)
