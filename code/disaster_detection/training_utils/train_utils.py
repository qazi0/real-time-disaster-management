import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from typing import Dict, Tuple
import logging
import numpy as np
from tqdm import tqdm

from .meters import AverageMeter

logger = logging.getLogger(__name__)

def train_epoch(
    config,
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # Metrics
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = Accuracy(task="multiclass", num_classes=config.num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=config.num_classes).to(device)
    
    # Progress bar
    progress = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Train Epoch: {epoch}/{config.epochs}")
    
    # Reset gradient accumulation counter
    optimizer.zero_grad()
    
    end = time.time()
    for batch_idx, (data, target) in progress:
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_amp):
            output = model(data)
            loss = criterion(output, target)
            
            # Scale loss by accumulation steps
            loss = loss / config.grad_accum_steps
        
        # Backward pass with gradient accumulation
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Clip gradients to prevent exploding gradients
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate
            if config.scheduler == "onecycle":
                if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scheduler.step()
        
        # Update metrics
        losses.update(loss.item() * config.grad_accum_steps, data.size(0))
        with torch.no_grad():
            preds = torch.argmax(output, dim=1)
            accuracy.update(preds, target)
            f1.update(preds, target)
        
        # Update progress bar
        progress.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc': f"{accuracy.compute().item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    # Compute final metrics
    metrics = {
        'train_loss': losses.avg,
        'train_acc': accuracy.compute().item(),
        'train_f1': f1.compute().item(),
        'lr': optimizer.param_groups[0]['lr']
    }
    
    # Log metrics
    logger.info(f"Train Epoch: {epoch} "
                f"Loss: {metrics['train_loss']:.4f} "
                f"Acc: {metrics['train_acc']:.4f} "
                f"F1: {metrics['train_f1']:.4f} "
                f"LR: {metrics['lr']:.6f}")
    
    return metrics

def validation_epoch(
    config,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    
    # Metrics
    losses = AverageMeter('Loss', ':.4e')
    accuracy = Accuracy(task="multiclass", num_classes=config.num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=config.num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=config.num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=config.num_classes).to(device)
    
    # Progress bar
    progress = tqdm(enumerate(val_loader), total=len(val_loader), 
                    desc=f"Val Epoch: {epoch}/{config.epochs}")
    
    with torch.no_grad():
        for batch_idx, (data, target) in progress:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            preds = torch.argmax(output, dim=1)
            accuracy.update(preds, target)
            f1.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{accuracy.compute().item():.4f}"
            })
    
    # Compute final metrics
    metrics = {
        'val_loss': losses.avg,
        'val_acc': accuracy.compute().item(),
        'val_f1': f1.compute().item(),
        'val_precision': precision.compute().item(),
        'val_recall': recall.compute().item()
    }
    
    # Log metrics
    logger.info(f"Validation Epoch: {epoch} "
                f"Loss: {metrics['val_loss']:.4f} "
                f"Acc: {metrics['val_acc']:.4f} "
                f"F1: {metrics['val_f1']:.4f} "
                f"Precision: {metrics['val_precision']:.4f} "
                f"Recall: {metrics['val_recall']:.4f}")
    
    return metrics

def test_epoch(
    config,
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Test model on test set."""
    model.eval()
    
    # Metrics
    losses = AverageMeter('Loss', ':.4e')
    accuracy = Accuracy(task="multiclass", num_classes=config.num_classes).to(device)
    f1 = F1Score(task="multiclass", num_classes=config.num_classes).to(device)
    precision = Precision(task="multiclass", num_classes=config.num_classes).to(device)
    recall = Recall(task="multiclass", num_classes=config.num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=config.num_classes).to(device)
    
    # Progress bar
    progress = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    
    # Collect all predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in progress:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            preds = torch.argmax(output, dim=1)
            accuracy.update(preds, target)
            f1.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            confmat.update(preds, target)
            
            # Collect predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{accuracy.compute().item():.4f}"
            })
    
    # Compute confusion matrix
    cm = confmat.compute().cpu().numpy()
    
    # Compute per-class metrics
    per_class_precision = np.diag(cm) / np.sum(cm, axis=0)
    per_class_recall = np.diag(cm) / np.sum(cm, axis=1)
    per_class_f1 = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall)
    
    # Class names
    class_names = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    logger.info('-' * 50)
    for i, name in enumerate(class_names):
        logger.info(f"{name:<20} {per_class_precision[i]:.4f}     {per_class_recall[i]:.4f}     {per_class_f1[i]:.4f}")
    logger.info('-' * 50)
    
    # Log confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(cm)
    
    # Compute final metrics
    metrics = {
        'test_loss': losses.avg,
        'test_acc': accuracy.compute().item(),
        'test_f1': f1.compute().item(),
        'test_precision': precision.compute().item(),
        'test_recall': recall.compute().item()
    }
    
    # Log metrics
    logger.info(f"\nTest Results: "
                f"Loss: {metrics['test_loss']:.4f} "
                f"Acc: {metrics['test_acc']:.4f} "
                f"F1: {metrics['test_f1']:.4f} "
                f"Precision: {metrics['test_precision']:.4f} "
                f"Recall: {metrics['test_recall']:.4f}")
    
    return metrics 