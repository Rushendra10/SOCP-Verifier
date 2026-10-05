from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(data_dir: str, batch_size: int, test_batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),  # maps pixels to [0,1]
    ])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    test_batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 training and test data loaders.

    Dataset:
        - Input size: 3 x 32 x 32
        - Number of classes: 10

    Images are converted directly to floats in [0,1] so the existing
    certification code (IBP, CROWN, Wong-Kolter, SOCP) can be used
    without modification.
    Did not use the standard mean and standard deviation to make it compatible with all other components
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader