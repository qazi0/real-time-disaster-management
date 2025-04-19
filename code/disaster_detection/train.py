import os
import sys
import time
import string
import uuid
import logging
import random
from dataclasses import asdict
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

from model.ernet import ErNET
from model.squeeze_ernet import Squeeze_ErNET
from model.squeeze_ernet_redconv import Squeeze_RedConv
from model.label_smoothing import LabelSmoothingCrossEntropy
from model.focal_loss import FocalLoss
from dataloaders.aider import AIDER, create_data_loaders, worker_init_fn
from training_utils import (
    TrainingConfig,
    EarlyStopping,
    AverageMeter,
    plot_training_curves,
    train_epoch,
    validation_epoch,
    test_epoch,
    parse_args
)

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
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(
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
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler == "reduce":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=config.epochs // 3,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

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
    if config.loss == 'label_smoothing_ce':
        criterion = LabelSmoothingCrossEntropy(
            epsilon=config.label_smoothing,
            reduction='mean'
        )
        logger.info(f"Using Label Smoothing Cross Entropy Loss with smoothing factor: {config.label_smoothing}")
    elif config.loss == 'focal':
        # Calculate class weights for focal loss
        class_counts = train_loader.dataset.annotations['label'].value_counts().sort_index()
        total_samples = len(train_loader.dataset.annotations)
        class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts])
        class_weights = class_weights / class_weights.sum()  # Normalize weights
        class_weights = class_weights.to(device)
        
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        logger.info(f"Using Focal Loss with class weights: {class_weights.tolist()}")
    else:
        raise ValueError(f"Unsupported loss function: {config.loss}")
    
    # Initialize optimizer
    optimizer = get_optimizer(config, model)
    
    # Initialize learning rate scheduler
    scheduler = get_scheduler(config, optimizer, train_loader)
    
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
