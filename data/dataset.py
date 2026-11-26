"""
ImageNet100 Dataset Loader
ImageNet100 is a subset of ImageNet containing 100 classes
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import json


class ImageNet100Dataset(Dataset):
    """ImageNet100 Dataset"""
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: Dataset root directory, should contain train.X*/ and val.X/ subdirectories
            split: 'train' or 'val'
            transform: Image transformations
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Determine data directories based on split
        if split == 'train':
            # Training data is distributed across multiple train.X* directories
            self.data_dirs = sorted([
                os.path.join(root, d) for d in os.listdir(root)
                if d.startswith('train.X') and os.path.isdir(os.path.join(root, d))
            ])
        elif split == 'val':
            # Validation data is in val.X directory
            val_dir = os.path.join(root, 'val.X')
            self.data_dirs = [val_dir] if os.path.exists(val_dir) else []
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load data paths and labels from multiple directories"""
        if not self.data_dirs:
            raise RuntimeError(f"No dataset directories found for split: {self.split}")
        
        # Collect all classes from all data directories
        all_classes = set()
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                classes = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
                all_classes.update(classes)
        
        classes = sorted(list(all_classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.classes = classes
        
        # Load images from all data directories
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory not found: {data_dir}")
                continue
            
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
                        self.targets.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images from {self.split} set")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Data directories: {self.data_dirs}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target
    
    def get_class_names(self):
        """Return list of class names"""
        return self.classes


def get_cifar_transforms(dataset_name='cifar10', img_size=32, split='train'):
    dataset_key = dataset_name.lower()
    stats = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    }
    if dataset_key not in stats:
        raise ValueError(f"Unsupported CIFAR dataset: {dataset_name}")
    mean, std = stats[dataset_key]
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_train_transforms(img_size=224):
    """Data augmentation for training"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=224):
    """Data preprocessing for validation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_dataloaders(data_root, batch_size=256, num_workers=4, img_size=224, pin_memory=True, dataset_name='imagenet100'):
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Dataset root directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        pin_memory: Whether to use pin memory for dataloader
        dataset_name: Name of the dataset ('imagenet100', 'cifar10', or 'cifar100')
        
    Returns:
        train_loader, val_loader, num_classes
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ('cifar10', 'cifar100'):
        dataset_cls = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
        # Create CIFAR datasets
        train_dataset = dataset_cls(
            root=data_root,
            train=True,
            download=True,
            transform=get_cifar_transforms(dataset_name, img_size, split='train')
        )
        
        val_dataset = dataset_cls(
            root=data_root,
            train=False,
            download=True,
            transform=get_cifar_transforms(dataset_name, img_size, split='val')
        )
        num_classes = 10 if dataset_name == 'cifar10' else 100
    else:
        # Create ImageNet100 datasets
        # Check if imagenet100 subdirectory exists and contains data
        imagenet_root = os.path.join(data_root, 'imagenet100')
        dataset_root = data_root
        
        if os.path.exists(imagenet_root) and os.path.isdir(imagenet_root):
            # Check if it contains training data folders
            if any(d.startswith('train.X') for d in os.listdir(imagenet_root)):
                dataset_root = imagenet_root
                
        train_dataset = ImageNet100Dataset(
            root=dataset_root,
            split='train',
            transform=get_train_transforms(img_size)
        )
        
        val_dataset = ImageNet100Dataset(
            root=dataset_root,
            split='val',
            transform=get_val_transforms(img_size)
        )
        num_classes = len(train_dataset.classes)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, num_classes


if __name__ == '__main__':
    # Test dataset
    import sys
    
    # Expected directory structure:
    # data_root/
    #   train.X1/
    #     class_001/
    #       img1.jpg
    #       img2.jpg
    #     class_002/
    #       ...
    #   train.X2/
    #     class_001/
    #       ...
    #   train.X3/
    #     ...
    #   train.X4/
    #     ...
    #   val.X/
    #     class_001/
    #       ...
    
    # Test code (requires actual data path)
    data_root = "/path/to/imagenet100"  # Replace with actual path
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    
    if os.path.exists(data_root):
        train_loader, val_loader, num_classes = get_dataloaders(
            data_root, batch_size=32, num_workers=2
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Num classes: {num_classes}")
        
        # Test one batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break
    else:
        print(f"Data root not found: {data_root}")
        print("Please provide a valid ImageNet100 dataset path")
