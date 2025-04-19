import os
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def plot_training_curves(metrics: Dict[str, List[float]], log_dir: str, model_name: str) -> None:
    """Plot training curves.
    
    Args:
        metrics: Dictionary of metrics
        log_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{model_name}_loss.png"))
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epoch'], metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{model_name}_accuracy.png"))
    plt.close()
    
    # Plot training and validation F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epoch'], metrics['train_f1'], label='Train F1')
    plt.plot(metrics['epoch'], metrics['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} - F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{model_name}_f1.png"))
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epoch'], metrics['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'{model_name} - Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{model_name}_lr.png"))
    plt.close()
    
    logger.info(f"Training plots saved to {log_dir}") 