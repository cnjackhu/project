import os
import torch
import torch.nn as nn
from utils import test_loop
from ratefunctiontorch import RateCumulant
from torch.nn import CrossEntropyLoss
from utils import zero_one_loss, get_losses
import pandas as pd

def get_argument_config():
    return {
        'dataset': {
            'type': str,
            'choices': ['cifar10', 'mnist', 'fashion'],
            'default': 'cifar10',
            'help': 'Dataset to use: cifar10, mnist, or fashion'
        },
        'epochs': {
            'type': int,
            'default': 200,
            'help': 'Number of epochs'
        },
        'lr': {
            'type': float,
            'default': 0.1,
            'help': 'learning rate'
        },
        'optimizer': {
            'type': str,
            'default': 'SGD',
            'help': 'optimizer'  # SGD or Adam
        },
        'batch_size': {
            'type': int,
            'default': 128,
            'help': 'Training batch size'
        },
        'batch_size_val': {
            'type': int,
            'default': 128,
            'help': 'Validation/Test batch size'
        },
        'buffer_size': {
            'type': int,
            'default': 2000,
            'help': 'Buffer size for online cumulant (defaults to batch-size-val)'
        },
        'forgetting_factor': {
            'type': float,
            'default': 1.0,
            'help': 'Forgetting factor for online cumulant'
        },
        'model_type': {
            'type': str,
            'default': 'ResNet18',
            'help': 'Model type'
        },
        'train_split': {
            'type': float,
            'default': 0.8,
            'help': 'Fraction of training data to use for training (remainder used for validation)'
        },
        'data_augmentation': {
            'type': str,
            'default': 'False',
            'help': 'Enable data augmentation for validation'
        },
        'reg': {
            'type': int,
            'default': 0,
            'help': 'Regularization type: 0=none'
        },
        'lamb': {
            'type': float,
            'default': 0.0,
            'help': 'Regularization parameter'
        }
    }

def init_simple_metrics():
    return {'acc_test':[],
    'L_test': [],
    'Loss_train': [],
    'acc_train':[],
    'it': []
    }

def update_simple_metrics(test_metrics, model, train_loader, test_loader,it):
    test_metrics['it'].append(it)
    test_loss,test_acc = test_loop(test_loader,model) 
    train_loss,train_acc = test_loop(train_loader,model) 
    test_metrics['acc_test'].append(test_acc)
    test_metrics['L_test'].append(test_loss)
    test_metrics['Loss_train'].append(train_loss)
    test_metrics['acc_train'].append(train_acc)
    return {
    'Train cce': test_metrics['Loss_train'][-1],
    'acc_train': test_metrics['acc_train'][-1],
    'L_test': test_metrics['L_test'][-1],
    'acc_test':test_metrics['acc_test'][-1]
     } 


def init_metrics():
    return {
        'L': [],
        'alphaD': [],
        'var': [],
        'lambda': [],
        'cummulant': [],
        'error': []
    }

@torch.no_grad()
def update_metrics(metrics_dic, model, loader, train_loss, loss_fn=nn.CrossEntropyLoss(reduction="none")):
    """
    Update metrics dictionary with current model performance metrics.
    """
    log_loss, zero_one_losses = get_losses(model, loader, loss_fn=[loss_fn, zero_one_loss])
    
    ratecumulant = RateCumulant.from_losses(log_loss)

    L = ratecumulant.compute_mean()
    alphaD, lamb, cummulant = ratecumulant.compute_rate_function(
        L - train_loss, return_lambdas=True, return_cummulants=True
    )
    var = ratecumulant.compute_variance()
    error = torch.mean(zero_one_losses).item()

    metrics_dic['L'].append(L.item())
    metrics_dic['alphaD'].append(alphaD.item())
    metrics_dic['var'].append(var.item())
    metrics_dic['lambda'].append(lamb.item())
    metrics_dic['cummulant'].append(cummulant.item())
    metrics_dic['error'].append(error)

@torch.no_grad()
def update_metrics_online(metrics_dic, ratecumulant, train_loss):
    """
    Update metrics dictionary with current model performance metrics.
    """
    L = ratecumulant.compute_mean()
    alphaD, lamb, cummulant = ratecumulant.compute_rate_function(
        L - train_loss, return_lambdas=True, return_cummulants=True
    )

    #rough estimate of error
    error = 1-torch.mean((torch.exp(-ratecumulant.get_losses())>0.5).float()).item()
    var = ratecumulant.compute_variance()

    metrics_dic['L'].append(L.item())
    metrics_dic['alphaD'].append(alphaD.item())
    metrics_dic['var'].append(var.item())
    metrics_dic['lambda'].append(lamb.item())
    metrics_dic['cummulant'].append(cummulant.item())
    metrics_dic['error'].append(error)

def save_metrics(metrics, args, folder_name='./data/metrics/', prefix=''):
    # Get the argument configuration
    arg_config = get_argument_config()
    
    # Start with prefix and metrics
    filename_parts = [f"{prefix}_metrics_"]
    
    # Add each argument value to filename if it exists in args
    for arg_name in arg_config.keys():
        if hasattr(args, arg_name):
            filename_parts.append(f"__{arg_name}_{getattr(args, arg_name)}")
    
    # Join all parts and add extension
    metrics_filename = "".join(filename_parts) + ".csv"
    
    os.makedirs(folder_name, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(f'{folder_name}/{metrics_filename}',float_format='%9.4f', index=False) 

