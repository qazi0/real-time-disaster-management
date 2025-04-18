import os
import sys
import time
import shutil
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
import datetime
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchmetrics
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from model.label_smoothing import LabelSmoothingCrossEntropy
from model.ernet import ErNET
from model.squeeze_ernet import Squeeze_ErNET
from model.squeeze_ernet_redconv import Squeeze_RedConv
from dataloaders.aider import AIDER, create_data_loaders, worker_init_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Still use benchmark for speed
import torchsummary
import argparse
import string
import uuid

from training_utils import (
    TrainingConfig,
    EarlyStopping,
    AverageMeter,
    plot_training_curves,
    train_epoch,
    validation_epoch,
    test_epoch
)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model settings
    model: str = "ernet"  # Options: "ernet", "squeeze-ernet", "squeeze-redconv"
    pretrained: bool = False
    resume: bool = False
    weights: Optional[str] = None
    summary: bool = False
    
    # Data settings
    root_dir: str = "data/AIDER"
    train_split: str = "dataloaders/aider_train.csv"
    val_split: str = "dataloaders/aider_val.csv"
    test_split: str = "dataloaders/aider_test.csv"
    image_size: int = 240
    use_albumentations: bool = True
    num_classes: int = 5
    
    # Dataloader settings
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Training settings
    epochs: int = 100
    optimizer: str = "adamw"  # Options: "adam", "adamw", "sgd"
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    momentum: float = 0.9
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    
    # Learning rate scheduler settings
    scheduler: str = "onecycle"  # Options: "onecycle", "cosine"
    warmup_epochs: int = 5
    warmup_ratio: float = 0.1
    
    # Regularization settings
    dropout: float = 0.2
    augment: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    
    # Mixed precision settings
    use_amp: bool = True
    
    # Checkpointing settings
    checkpoint_dir: str = "saves"
    checkpoint_freq: int = 1
    save_best_only: bool = True
    
    # Early stopping settings
    early_stopping: bool = True
    patience: int = 10
    
    # Misc settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    debug: bool = False
    log_dir: str = "logs"
    collab: bool = False
    
    def __post_init__(self):
        """Initialize derived settings."""
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set path for saving model
        if self.weights is None:
            self.weights = os.path.join(self.checkpoint_dir, f"{self.model}.pt")
        
        # Infer image size from model
        if self.model == "ernet":
            self.image_size = 240
        else:
            self.image_size = 140
            
        # Adjust batch size for smaller models
        if self.model != "ernet":
            self.batch_size *= 2
            
        # Save config to JSON
        self.save_config()
            
    def save_config(self):
        """Save config to JSON file."""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
        logger.info(f"Config saved to {config_path}")
        

def init_weights(m: nn.Module):
    """Initialize model weights for better convergence."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_optimizer(config: TrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Get optimizer based on config."""
    if config.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_scheduler(config: TrainingConfig, optimizer: torch.optim.Optimizer, train_loader: DataLoader) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler based on config."""
    if config.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.warmup_ratio,
            div_factor=25.0,
            final_div_factor=10000.0,
            anneal_strategy='cos'
        )
    elif config.scheduler == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.epochs // 3,
            T_mult=2,
            eta_min=config.min_lr
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def generate_random_string(length: int = 6) -> str:
    """Generate a random string for model naming."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def train_model(config: TrainingConfig) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Main training function orchestrating the training process."""
    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # Set device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create data loaders with optimized settings
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv=config.train_split,
        val_csv=config.val_split,
        test_csv=config.test_split,
        root_dir=config.root_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_albumentations=config.use_albumentations,
        image_size=config.image_size,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers
    )
    
    # Log dataset sizes
    logger.info(f"Training set: {len(train_loader.dataset)} samples")
    logger.info(f"Validation set: {len(val_loader.dataset)} samples")
    logger.info(f"Test set: {len(test_loader.dataset)} samples")
    
    # Initialize model
    logger.info(f"Initializing model: {config.model}")
    if config.model == 'ernet':
        model = ErNET()
    elif config.model == 'squeeze-ernet':
        model = Squeeze_ErNET()
    elif config.model == 'squeeze-redconv':
        model = Squeeze_RedConv()
    else:
        raise ValueError(f"Unsupported model: {config.model}")
    
    # Initialize model weights if not resuming
    if not config.resume and not config.pretrained:
        logger.info("Initializing model weights")
        model.apply(init_weights)
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary if requested
    if config.summary:
        input_size = (3, config.image_size, config.image_size)
        logger.info(f"Model summary (input size: {input_size}):")
        torchsummary.summary(model, input_size, device=str(device))
    
    # Initialize criterion (loss function)
    try:
        criterion = LabelSmoothingCrossEntropy(epsilon=config.label_smoothing)
        logger.info(f"Using label smoothing cross entropy with epsilon={config.label_smoothing}")
    except ValueError as e:
        logger.warning(f"Error initializing label smoothing: {e}. Using standard cross entropy.")
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    # Initialize optimizer
    optimizer = get_optimizer(config, model)
    
    # Initialize learning rate scheduler
    scheduler = get_scheduler(
        config, 
        optimizer, 
        train_loader
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.use_amp)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [], 
                    'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    
    if config.resume and os.path.exists(config.weights):
        logger.info(f"Loading checkpoint from {config.weights}")
        checkpoint = torch.load(config.weights, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Modern checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and config.use_amp:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            if 'train_metrics' in checkpoint:
                train_metrics = checkpoint['train_metrics']
        else:
            # Legacy format (just the model)
            model.load_state_dict(checkpoint)
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        verbose=True,
        path=os.path.join(config.checkpoint_dir, f"{config.model}_best.pt")
    )
    
    # Training loop
    logger.info(f"Starting training for {config.epochs} epochs")
    for epoch in range(start_epoch, config.epochs):
        # Train for one epoch
        train_epoch_metrics = train_epoch(
            config=config,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            device=device
        )
        
        # Validate
        val_epoch_metrics = validation_epoch(
            config=config,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            epoch=epoch,
            device=device
        )
        
        # Update scheduler if it's not OneCycle (which updates every step)
        if config.scheduler != "onecycle":
            scheduler.step(val_epoch_metrics['val_loss'])
        
        # Save metrics
        train_metrics['epoch'].append(epoch)
        train_metrics['train_loss'].append(train_epoch_metrics['train_loss'])
        train_metrics['train_acc'].append(train_epoch_metrics['train_acc'])
        train_metrics['train_f1'].append(train_epoch_metrics['train_f1'])
        train_metrics['val_loss'].append(val_epoch_metrics['val_loss'])
        train_metrics['val_acc'].append(val_epoch_metrics['val_acc'])
        train_metrics['val_f1'].append(val_epoch_metrics['val_f1'])
        train_metrics['lr'].append(train_epoch_metrics['lr'])
        
        # Check if current model is the best
        is_best = val_epoch_metrics['val_acc'] > best_val_acc
        if is_best:
            best_val_acc = val_epoch_metrics['val_acc']
            best_val_loss = val_epoch_metrics['val_loss']
            
        # Save checkpoint
        if epoch % config.checkpoint_freq == 0 or is_best or epoch == config.epochs - 1:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"{config.model}_{epoch:03d}.pt" if not is_best else f"{config.model}_best.pt"
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'scaler_state_dict': scaler.state_dict() if config.use_amp else None,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'config': asdict(config)
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # If current model is the best, save a copy
            if is_best and checkpoint_path != os.path.join(config.checkpoint_dir, f"{config.model}_best.pt"):
                best_path = os.path.join(config.checkpoint_dir, f"{config.model}_best.pt")
                shutil.copyfile(checkpoint_path, best_path)
                logger.info(f"Best model saved to {best_path}")
        
        # Early stopping
        if config.early_stopping:
            early_stopping(val_epoch_metrics['val_loss'], model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
    
    # Load the best model for testing
    best_model_path = os.path.join(config.checkpoint_dir, f"{config.model}_best.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # Test the model
    logger.info("Evaluating model on test set")
    test_metrics = test_epoch(
        config=config,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Plot training curves
    plot_training_curves(train_metrics, config.log_dir, config.model)
    
    return model, {**train_metrics, **test_metrics}


def parse_args() -> TrainingConfig:
    """Parse command line arguments and convert to TrainingConfig."""
    parser = argparse.ArgumentParser(description='PyTorch ErNet on AIDER Dataset')
    
    # Model settings
    parser.add_argument('--model', type=str, default='ernet',
                        choices=['ernet', 'squeeze-ernet', 'squeeze-redconv'],
                        help='model architecture: ernet, squeeze-ernet, squeeze-redconv (default: ernet)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the weights file (.pt) for resuming training or evaluation')
    parser.add_argument('--summary', action='store_true',
                        help='print model summary and exit')
                        
    # Data settings
    parser.add_argument('--root-dir', type=str, default='data/AIDER',
                        help='path to the root directory containing the AIDER dataset')
    parser.add_argument('--train-split', type=str, default='dataloaders/aider_train.csv',
                        help='path to the training split CSV file')
    parser.add_argument('--val-split', type=str, default='dataloaders/aider_val.csv',
                        help='path to the validation split CSV file')
    parser.add_argument('--test-split', type=str, default='dataloaders/aider_test.csv',
                        help='path to the test split CSV file')
    parser.add_argument('--image-size', type=int, default=None,
                        help='image size (determined automatically if not specified)')
    parser.add_argument('--no-albumentations', action='store_true',
                        help='disable Albumentations transforms and use torchvision transforms')
                        
    # Dataloader settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of worker threads for data loading (default: 8)')
    parser.add_argument('--no-pin-memory', action='store_true',
                        help='disable pin_memory in data loader')
                        
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer: adam, adamw, sgd (default: adamw)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='minimum learning rate (default: 1e-6)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing factor (default: 0.1)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping norm (default: 1.0)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='gradient accumulation steps (default: 1)')
                        
    # Learning rate scheduler settings
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['onecycle', 'cosine'],
                        help='learning rate scheduler: onecycle, cosine (default: onecycle)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='warmup ratio (default: 0.1)')
                        
    # Regularization settings
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout probability (default: 0.2)')
    parser.add_argument('--no-augment', action='store_true',
                        help='disable data augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='mixup alpha (default: 0.2)')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0,
                        help='cutmix alpha (default: 0.0)')
                        
    # Mixed precision settings
    parser.add_argument('--no-amp', action='store_true',
                        help='disable automatic mixed precision')
                        
    # Checkpointing settings
    parser.add_argument('--checkpoint-dir', type=str, default='saves',
                        help='directory to save checkpoints (default: saves)')
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='checkpoint frequency in epochs (default: 1)')
    parser.add_argument('--save-best-only', action='store_true',
                        help='save only the best model (based on validation accuracy)')
                        
    # Early stopping settings
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='disable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping (default: 10)')
                        
    # Misc settings
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory to save logs and plots (default: logs)')
    parser.add_argument('--collab', action='store_true',
                        help='use Google Colab paths')
                        
    args = parser.parse_args()
    
    # Create TrainingConfig from args
    config = TrainingConfig(
        # Model settings
        model=args.model,
        pretrained=args.pretrained,
        resume=args.resume,
        weights=args.weights,
        summary=args.summary,
        
        # Data settings
        root_dir=args.root_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size if args.image_size is not None else (240 if args.model == 'ernet' else 140),
        use_albumentations=not args.no_albumentations,
        
        # Dataloader settings
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        
        # Training settings
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        
        # Learning rate scheduler settings
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        warmup_ratio=args.warmup_ratio,
        
        # Regularization settings
        dropout=args.dropout,
        augment=not args.no_augment,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        
        # Mixed precision settings
        use_amp=not args.no_amp,
        
        # Checkpointing settings
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_best_only=args.save_best_only,
        
        # Early stopping settings
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        
        # Misc settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        debug=args.debug,
        log_dir=args.log_dir,
        collab=args.collab
    )
    
    return config


def main():
    """Main entry point for training."""
    try:
        # Parse arguments
        config = parse_args()
        
        # Log configuration
        logger.info(f"Starting training with configuration:")
        for key, value in asdict(config).items():
            logger.info(f"  {key}: {value}")
        
        # Exit after printing model summary if requested
        if config.summary:
            # Initialize model
            if config.model == 'ernet':
                model = ErNET()
            elif config.model == 'squeeze-ernet':
                model = Squeeze_ErNET()
            elif config.model == 'squeeze-redconv':
                model = Squeeze_RedConv()
            else:
                raise ValueError(f"Unsupported model: {config.model}")
                
            # Print model summary
            input_size = (3, config.image_size, config.image_size)
            device = torch.device(config.device)
            model = model.to(device)
            torchsummary.summary(model, input_size, device=str(device))
            return
        
        # Train model
        logger.info("Starting training")
        model, metrics = train_model(config)
        
        # Log final metrics
        logger.info("Training completed successfully")
        logger.info(f"Final validation accuracy: {metrics['val_acc'][-1]:.4f}")
        logger.info(f"Final validation F1 score: {metrics['val_f1'][-1]:.4f}")
        logger.info(f"Test accuracy: {metrics['test_acc']:.4f}")
        logger.info(f"Test F1 score: {metrics['test_f1']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
