import argparse
from typing import Optional
from .config import TrainingConfig
import torch

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
    parser.add_argument('--loss', type=str, default='label_smoothing_ce',
                        choices=['label_smoothing_ce', 'focal'],
                        help='loss function: label_smoothing_ce, focal (default: label_smoothing_ce)')
                        
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
    parser.add_argument('--checkpoint-freq', type=int, default=30,
                        help='checkpoint frequency in epochs (default: 30)')
    parser.add_argument('--save-best-only', action='store_true',
                        help='save only the best model (based on validation accuracy)')
                        
    # Early stopping settings
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='disable early stopping')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stopping (default: 50)')
                        
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
        loss=args.loss,
        
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