import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(data_path, batch):
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)
    return train_loader, test_loader
