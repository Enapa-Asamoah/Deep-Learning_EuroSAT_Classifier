import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data.dataset import TrashNetDataset

def create_dataloaders(data_dir, train_idx, val_idx, test_idx, batch_size=32, num_workers=4, pin_memory=True, transform_train=None, transform_val=None):
    """
    Creates PyTorch DataLoaders for train/val/test sets
    Args:
        data_dir (str or Path): Root directory of dataset
        train_idx, val_idx, test_idx (list[int]): Indices for splits
        batch_size (int): Batch size
        num_workers (int): Number of DataLoader workers
        pin_memory (bool): Pin memory flag
        transform_train (callable): Transform for training data
        transform_val (callable): Transform for val/test data
    Returns:
        train_loader, val_loader, test_loader
    """

    full_dataset = TrashNetDataset(data_dir)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Apply transforms
    if transform_train:
        train_dataset.dataset.transform = transform_train
    if transform_val:
        val_dataset.dataset.transform = transform_val
        test_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def create_splits(dataset_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Creates fixed train/val/test indices
    Returns: train_idx, val_idx, test_idx
    """
    indices = list(range(dataset_size))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=random_seed, stratify=None)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(test_ratio + val_ratio), random_state=random_seed, stratify=None)
    return train_idx, val_idx, test_idx