import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from functools import lru_cache
import logging
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""Aerial Imagery Dataset for Emergency Response

Dataset organization:
    - 5 classes: collapsed_building (0), fire (1), flooded_areas (2), normal (3), traffic_incident (4)
    - ~6433 total images with class imbalance (normal class has the most images)

Split files:
    - aider_labels.csv: Original labels file with all images
    - aider_train.csv: 70% of data, stratified by class 
    - aider_val.csv: 20% of data, stratified by class
    - aider_test.csv: 10% of data, stratified by class
"""

# Cache for loaded images to improve performance
@lru_cache(maxsize=1024)
def cached_image_loader(img_path: str) -> Image.Image:
    """Load image with caching for improved performance.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Loaded PIL Image
    """
    try:
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {img_path}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (240, 240), color=(0, 0, 0))


class AIDER(Dataset):
    """
    PyTorch Dataset for AIDER (Aerial Imagery Dataset for Emergency Response).
    
    Args:
        csv_file: Path to the CSV file with image paths and labels.
            Can be absolute path or relative to the current working directory.
            Pre-generated split files are available: aider_train.csv, aider_val.csv, aider_test.csv
        root_dir: Directory containing the images. Image paths in the CSV are relative to this.
        transform: Optional transform to be applied on the images.
        use_albumentations: Whether to use Albumentations library for transforms
            (faster and more memory efficient than torchvision)
        cache_size: Number of images to cache in memory
        image_size: Size to resize images to (default: 240)
        
    Attributes:
        annotations: DataFrame containing image paths and labels
        samples: List of (image_path, label) tuples
        class_weights: Weight for each class (for weighted sampling)
        mean: Dataset mean (calculated once and cached)
        std: Dataset standard deviation (calculated once and cached)
    """
    def __init__(
        self, 
        csv_file: Union[str, Path], 
        root_dir: Union[str, Path], 
        transform: Optional[Callable] = None,
        use_albumentations: bool = True,
        cache_size: int = 1024,
        image_size: int = 240,
        is_training: bool = False
    ):
        csv_file = Path(csv_file)
        root_dir = Path(root_dir)
        
        # If csv_file is not an absolute path, check in current directory and dataloaders directory
        if not csv_file.is_absolute():
            # First check if the file exists as specified
            if not csv_file.exists():
                # Then check in the dataloaders directory
                base_dir = Path(__file__).parent.absolute()
                alternative_path = base_dir / csv_file.name
                if alternative_path.exists():
                    csv_file = alternative_path
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        # Load annotations
        self.annotations = pd.read_csv(csv_file, header=None, names=['path', 'label'])
        
        # Store samples as a list of tuples for faster access
        self.samples = list(zip(
            [str(root_dir / path) for path in self.annotations['path']],
            self.annotations['label'].values
        ))
        
        # Store root directory and transform
        self.root_dir = root_dir
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.image_size = image_size
        self.is_training = is_training
        
        # Calculate class distribution
        self.class_counts = self.annotations['label'].value_counts().sort_index()
        total_samples = len(self.annotations)
        
        # Calculate class weights for weighted sampling
        num_classes = len(self.class_counts)
        self.class_weights = {
            label: total_samples / (num_classes * count) 
            for label, count in self.class_counts.items()
        }
        
        # Set up default transforms if none provided
        if transform is None:
            try:
                if use_albumentations:
                    logger.info(f"Using albumentations transforms (is_training={is_training})")
                    if is_training:
                        self.transform = get_train_albumentation_transforms(image_size)
                    else:
                        self.transform = get_val_albumentation_transforms(image_size)
                    
                    # If albumentations returned None (fallback case), use torchvision
                    if self.transform is None:
                        logger.warning("Albumentations failed, using torchvision transforms")
                        if is_training:
                            self.transform = get_train_torchvision_transforms(image_size)
                        else:
                            self.transform = get_val_torchvision_transforms(image_size)
                else:
                    logger.info(f"Using torchvision transforms (is_training={is_training})")
                    if is_training:
                        self.transform = get_train_torchvision_transforms(image_size)
                    else:
                        self.transform = get_val_torchvision_transforms(image_size)
            except Exception as e:
                logger.error(f"Error setting up transforms: {e}")
                logger.warning("Falling back to basic torchvision transforms")
                # Fall back to basic torchvision transforms if all else fails
                if is_training:
                    self.transform = get_train_torchvision_transforms(image_size)
                else:
                    self.transform = get_val_torchvision_transforms(image_size)
                
        # Calculate and cache dataset statistics if not already done
        self._calculate_stats()
        
        logger.info(f"Loaded AIDER dataset from {csv_file} with {len(self)} samples")
        logger.info(f"Class distribution: {self.class_counts.to_dict()}")

    def _calculate_stats(self) -> None:
        """Calculate dataset mean and standard deviation."""
        # Load cached stats if available
        cache_file = Path(__file__).parent / 'aider_stats.pt'
        if cache_file.exists():
            stats = torch.load(cache_file)
            self.mean = stats['mean']
            self.std = stats['std']
            logger.info(f"Loaded cached dataset statistics from {cache_file}")
            return
            
        logger.info("Calculating dataset statistics...")
        # Calculate from a subset for efficiency
        subset_size = min(1000, len(self))
        indices = np.random.choice(len(self), subset_size, replace=False)
        pixel_sum = torch.zeros(3)
        pixel_sum_squared = torch.zeros(3)
        pixel_count = 0
        
        # Use a simple transform just for computing stats
        temp_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        
        for idx in indices:
            img_path, _ = self.samples[idx]
            try:
                img = cached_image_loader(img_path)
                img_tensor = temp_transform(img)
                
                # Update stats
                pixel_sum += img_tensor.sum(dim=[1, 2])
                pixel_sum_squared += (img_tensor ** 2).sum(dim=[1, 2])
                pixel_count += img_tensor.shape[1] * img_tensor.shape[2]
            except Exception as e:
                logger.warning(f"Error processing image {img_path} for stats: {e}")
                
        # Calculate mean and std
        mean = pixel_sum / pixel_count
        std = torch.sqrt((pixel_sum_squared / pixel_count) - (mean ** 2))
        
        self.mean = mean.tolist()
        self.std = std.tolist()
        
        # Cache the stats
        torch.save({'mean': self.mean, 'std': self.std}, cache_file)
        logger.info(f"Dataset stats - mean: {self.mean}, std: {self.std}")
        logger.info(f"Saved dataset statistics to {cache_file}")
        
    def __len__(self) -> int:
        """Return the size of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            Tuple of (image, label) where image is a tensor and label is a tensor
        """
        img_path, label = self.samples[index]
        
        # Load image with caching
        img = cached_image_loader(img_path)
        
        # Apply transforms
        if self.transform:
            if self.use_albumentations:
                # Albumentations expects numpy array
                img_np = np.array(img)
                transformed = self.transform(image=img_np)
                img_tensor = transformed["image"]
            else:
                img_tensor = self.transform(img)
        else:
            # Fallback transformation
            img_tensor = transforms.ToTensor()(img)
            
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

    def get_sample_weights(self) -> List[float]:
        """Get sample weights for weighted random sampling.
        
        Returns:
            List of weights for each sample
        """
        return [self.class_weights.get(label, 1.0) for _, label in self.samples]
def get_train_albumentation_transforms(image_size: int = 240) -> A.Compose:
    """Get training transforms using Albumentations.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Albumentations transform composition
    """
    logger.info(f"Creating training transforms with image_size={image_size}")
    
    # Define a function for the standard transform pipeline
    def get_standard_transform_pipeline():
        return [
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.2),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
    # Try different approaches with proper error handling
    try:
        # Method 1: Try with single integer size parameter (newest API)
        try:
            transform_list = [A.RandomResizedCrop(size=image_size, scale=(0.8, 1.0))]
            transform_list.extend(get_standard_transform_pipeline())
            return A.Compose(transform_list)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error creating transforms with 'size=image_size': {e}")
            logger.warning("Trying with size as tuple format...")
            
            # Method 2: Try with tuple format for size parameter
            try:
                transform_list = [A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0))]
                transform_list.extend(get_standard_transform_pipeline())
                return A.Compose(transform_list)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error creating transforms with 'size=(h,w)' format: {e}")
                logger.warning("Falling back to height/width parameters...")
                
                # Method 3: Try with separate height/width parameters (older API)
                try:
                    transform_list = [A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0))]
                    transform_list.extend(get_standard_transform_pipeline())
                    return A.Compose(transform_list)
                except Exception as e:
                    logger.error(f"Error with height/width parameters: {e}")
                    logger.warning("Falling back to minimal transforms...")
                    
                    # Method 4: Minimal transforms without RandomResizedCrop
                    return A.Compose([
                        A.Resize(height=image_size, width=image_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
    except Exception as e:
        # Last resort fallback - absolute minimal transforms
        logger.error(f"Critical error creating transforms: {e}")
        logger.warning("Falling back to absolute minimal transforms...")
        try:
            return A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        except Exception as e:
            # If even minimal transforms fail, fall back to torchvision
            logger.error(f"Fatal error with albumentations: {e}")
            logger.warning("Falling back to torchvision transforms")
            return None


def get_val_albumentation_transforms(image_size: int = 240) -> A.Compose:
    """Get validation transforms using Albumentations.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Albumentations transform composition
    """
    try:
        logger.info(f"Creating validation transforms with image_size={image_size}")
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    except Exception as e:
        # Catch any errors and fall back to even simpler transforms
        logger.error(f"Error creating validation transforms: {e}")
        logger.warning("Falling back to minimal validation transforms")
        try:
            return A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        except Exception as e:
            logger.error(f"Critical error with albumentations: {e}")
            logger.warning("Falling back to torchvision transforms")
            return None


def get_train_torchvision_transforms(image_size: int = 240) -> transforms.Compose:
    """Get training transforms using torchvision.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Torchvision transform composition
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_torchvision_transforms(image_size: int = 240) -> transforms.Compose:
    """Get validation transforms using torchvision.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Torchvision transform composition
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 1.14 ~ 256/224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Global variables for transforms with common defaults
aider_transforms = get_val_torchvision_transforms(240)
squeeze_transforms = get_val_torchvision_transforms(140)


def worker_init_fn(worker_id: int) -> None:
    """Initialize the worker process with a different random seed.
    
    Args:
        worker_id: ID of the worker process
    """
    # Set the numpy seed for this worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_data_loaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Union[str, Path],
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    use_albumentations: bool = True,
    image_size: int = 240,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        train_csv: Path to the training CSV file
        val_csv: Path to the validation CSV file
        test_csv: Path to the testing CSV file
        root_dir: Directory containing the images
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        use_albumentations: Whether to use Albumentations for transforms
        image_size: Size to resize images to
        pin_memory: Whether to pin memory in data loader
        prefetch_factor: Number of samples to prefetch per worker
        persistent_workers: Whether to keep worker processes alive after iterator exhaustion
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders with batch_size={batch_size}, num_workers={num_workers}")
    
    # Create datasets
    train_dataset = AIDER(
        csv_file=train_csv,
        root_dir=root_dir,
        use_albumentations=use_albumentations,
        image_size=image_size,
        is_training=True
    )
    
    val_dataset = AIDER(
        csv_file=val_csv,
        root_dir=root_dir,
        use_albumentations=use_albumentations,
        image_size=image_size,
        is_training=False
    )
    
    test_dataset = AIDER(
        csv_file=test_csv,
        root_dir=root_dir,
        use_albumentations=use_albumentations,
        image_size=image_size,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    logger.info(f"Created data loaders with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader
