import os
import time
import torch
import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from dataloaders.aider import AIDER, create_data_loaders, worker_init_fn
from model.ernet import ErNET
from model.squeeze_ernet import Squeeze_ErNET
from model.squeeze_ernet_redconv import Squeeze_RedConv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_name: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load model and weights."""
    # Initialize model based on name
    if model_name == 'ernet':
        model = ErNET()
    elif model_name == 'squeeze-ernet':
        model = Squeeze_ErNET()
    elif model_name == 'squeeze-redconv':
        model = Squeeze_RedConv()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Modern checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Legacy format (just the model)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_trt: bool = False,
    quant: str = 'fp16'
) -> Dict[str, float]:
    """Evaluate model on test set."""
    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=5).to(device)
    f1 = F1Score(task="multiclass", num_classes=5).to(device)
    precision = Precision(task="multiclass", num_classes=5).to(device)
    recall = Recall(task="multiclass", num_classes=5).to(device)
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=5).to(device)
    
    # Initialize timing metrics
    inference_times = []
    
    # Evaluate
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            if use_trt and quant == 'fp16':
                data = data.half()
            
            # Time inference
            start_time = time.time()
            output = model(data)
            torch.cuda.synchronize()  # Wait for GPU to finish
            inference_times.append(time.time() - start_time)
            
            # Update metrics
            pred = output.argmax(dim=1)
            accuracy.update(pred, target)
            f1.update(pred, target)
            precision.update(pred, target)
            recall.update(pred, target)
            confusion_matrix.update(pred, target)
    
    # Compute final metrics
    metrics = {
        'accuracy': accuracy.compute().item(),
        'f1_score': f1.compute().item(),
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'avg_inference_time': np.mean(inference_times),
        'fps': 1.0 / np.mean(inference_times)
    }
    
    # Compute per-class metrics from confusion matrix
    cm = confusion_matrix.compute()
    per_class_metrics = compute_per_class_metrics(cm)
    metrics.update(per_class_metrics)
    
    return metrics

def compute_per_class_metrics(confusion_matrix: torch.Tensor) -> Dict[str, float]:
    """Compute per-class metrics from confusion matrix."""
    metrics = {}
    classes = ['collapsed building', 'fire', 'flooded areas', 'normal', 'traffic incident']
    
    for i, class_name in enumerate(classes):
        # True positives
        tp = confusion_matrix[i, i].item()
        # False positives
        fp = confusion_matrix[:, i].sum().item() - tp
        # False negatives
        fn = confusion_matrix[i, :].sum().item() - tp
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.update({
            f'{class_name}_precision': precision,
            f'{class_name}_recall': recall,
            f'{class_name}_f1': f1
        })
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model', type=str, default='ernet',
                        choices=['ernet', 'squeeze-ernet', 'squeeze-redconv'],
                        help='model architecture')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--test-split', type=str, default='dataloaders/aider_test.csv',
                        help='path to test split CSV')
    parser.add_argument('--root-dir', type=str, default='data/AIDER',
                        help='path to dataset root directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of worker threads for data loading')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--trt', action='store_true',
                        help='use TensorRT for inference')
    parser.add_argument('--quant', type=str, default='fp16',
                        choices=['fp16', 'fp32'],
                        help='quantization scheme for TensorRT')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create test loader
    _, _, test_loader = create_data_loaders(
        train_csv=None,
        val_csv=None,
        test_csv=args.test_split,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_albumentations=False,  # No augmentation for evaluation
        image_size=240 if args.model == 'ernet' else 140,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Load model
    model = load_model(args.model, args.weights, device)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, args.trt, args.quant)
    
    # Print results
    logger.info("\nEvaluation Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Average Inference Time: {metrics['avg_inference_time']:.4f} seconds")
    logger.info(f"FPS: {metrics['fps']:.2f}")
    
    logger.info("\nPer-class Metrics:")
    for class_name in ['collapsed building', 'fire', 'flooded areas', 'normal', 'traffic incident']:
        logger.info(f"\n{class_name}:")
        logger.info(f"  Precision: {metrics[f'{class_name}_precision']:.4f}")
        logger.info(f"  Recall: {metrics[f'{class_name}_recall']:.4f}")
        logger.info(f"  F1 Score: {metrics[f'{class_name}_f1']:.4f}")

if __name__ == '__main__':
    main()
