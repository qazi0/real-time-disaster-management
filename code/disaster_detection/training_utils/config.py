import os
import json
import logging
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model settings
    model: str = "squeeze-ernet"  # Changed default to squeeze-ernet
    pretrained: bool = False
    resume: bool = False
    weights: Optional[str] = None
    summary: bool = False
    loss: str = 'label_smoothing_ce'  # Add loss function choice
    
    # Data settings
    root_dir: str = "data/AIDER"
    train_split: str = "dataloaders/aider_train.csv"
    val_split: str = "dataloaders/aider_val.csv"
    test_split: str = "dataloaders/aider_test.csv"
    image_size: int = 240
    use_albumentations: bool = True
    num_classes: int = 5
    
    # Dataloader settings
    batch_size: int = 32  # Reduced batch size for better generalization
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Training settings
    epochs: int = 200  # 
    optimizer: str = "adamw"  # Using AdamW with weight decay
    lr: float = 3e-4  # learning rate
    min_lr: float = 1e-6
    weight_decay: float = 0.01 
    momentum: float = 0.9
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 2 
    
    # Learning rate scheduler settings
    scheduler: str = "onecycle"  
    warmup_epochs: int = 5
    warmup_ratio: float = 0.1
    
    # Regularization settings
    dropout: float = 0.2  # Increased dropout
    augment: bool = True
    mixup_alpha: float = 0.2 
    cutmix_alpha: float = 0.1  
    
    # Mixed precision settings
    use_amp: bool = True
    
    # Checkpointing settings
    checkpoint_dir: str = "saves"
    checkpoint_freq: int = 1
    save_best_only: bool = True
    
    # Early stopping settings
    early_stopping: bool = True
    patience: int = 20  # Increased patience
    
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