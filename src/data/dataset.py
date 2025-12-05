"""
Data loading and preprocessing for Fashion-MNIST.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_data(batch_size=128, data_root='./data', num_workers=2):
    """
    Load and preprocess Fashion-MNIST dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        data_root: Root directory for data storage
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    # Normalize to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.FashionMNIST(
        root=data_root, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_root, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


# Fashion-MNIST class names for reference
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]