import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn
import os
import numpy as np
from models import *



def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

RANDOM_SEED = 12345678
set_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_transforms(dataset,augmentation='False'):
    if dataset == 'cifar10':
        if augmentation=='True':
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    else:  # mnist
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform_train, transform_test

def load_datasets(dataset, transform_train,transform_test,train_split=0.8):
    if dataset == 'cifar10':
        full_trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif dataset == 'fashion':
        full_trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
    else:  # mnist
        full_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    
    # Create train/val split
    train_size = int(train_split * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
        
    return trainset, valset, testset

def create_model(model_type, device):
    if model_type == "ResNet18":
        print("load resnet18...")
        return ResNet18().to(device)
    elif model_type == "cnn":
        print("load mnistnet...")
        return MNISTNet().to(device)
    elif model_type == "mlp":
        print("load mlp...")
        return MNISTMLP().to(device)


def create_dataloaders(trainset, valset, testset, batch_size, batch_size_val):
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size_val,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader



def test_loop(dataloader, model, loss_fn=nn.CrossEntropyLoss()):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # code from pytorch website:https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    model.eval()
    device = get_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, (100*correct)


def zero_one_loss(logits, targets):
    """
    Compute 0-1 loss (misclassification error) for each sample.
    
    Parameters
    ----------
    logits : torch.Tensor
        Model output logits
    targets : torch.Tensor
        Ground truth labels
        
    Returns
    -------
    torch.Tensor
        0-1 loss for each sample (1 for misclassification, 0 for correct)
    """
    predictions = torch.argmax(logits, dim=1)
    return (predictions != targets).float()


def get_losses(model, loader, loss_fn=nn.CrossEntropyLoss(reduction="none")):
    """
    Compute one or multiple losses for a given model using data from a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        A DataLoader providing the dataset to evaluate.
    loss_fn : callable or list of callable, optional
        Single loss function or list of loss functions to evaluate.
        Default is nn.CrossEntropyLoss with no reduction.
        
    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        If single loss_fn: Tensor containing the losses for each batch
        If multiple loss_fns: Tuple of tensors, one for each loss function
    """
    # Get a parameter from the model to determine its device (GPU/CPU)
    p = next(model.parameters())
    device = p.device

    # Convert single loss function to list for unified processing
    if not isinstance(loss_fn, (list, tuple)):
        loss_fn = [loss_fn]
    
    # Initialize list of losses for each function
    all_losses = [[] for _ in loss_fn]
    
    # Set the model to evaluation mode
    model.eval()
    
    # Loop over the data loader
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(data)
        
        # Compute all losses
        for i, fn in enumerate(loss_fn):
            loss = fn(logits, targets)
            all_losses[i].append(loss)
    
    # Concatenate losses for each function
    all_losses = [torch.cat(losses) for losses in all_losses]
    
    # Return single tensor if only one loss function, otherwise return tuple
    return all_losses[0] if len(all_losses) == 1 else tuple(all_losses)
