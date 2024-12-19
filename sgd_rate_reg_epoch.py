import sys
import os
import numpy as np
from itertools import cycle
from ratefunctiontorch import OnlineCumulant, RateCumulant
from utils import get_device,set_seed, get_transforms, load_datasets, create_model, \
create_dataloaders, init_metrics,init_simple_metrics, save_metrics, update_metrics,update_simple_metrics
import tabulate
import torch
from torch import nn
from tqdm import tqdm
import argparse
import time
from models import *
# Keep existing argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['cifar10', 'mnist', 'fashion'], 
                    default='cifar10', help='Dataset to use: cifar10, mnist, or fashion')
parser.add_argument('--epochs', type=int, default=200,
                   help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.1,
                   help='learning rate')
parser.add_argument('--optimizer', type=str, default='SGD',
                   help='optimizer')  # SGD or Adam
parser.add_argument('--batch-size', type=int, default=128,
                   help='Training batch size')
parser.add_argument('--batch-size-val', type=int, default=100,
                   help='Validation/Test batch size')
parser.add_argument('--buffer-size', type=int, default=2000,
                   help='Buffer size for online cumulant (defaults to batch-size-val)')
parser.add_argument('--forgetting-factor', type=float, default=1.0,
                   help='Forgetting factor for online cumulant')

parser.add_argument('--model-type', type=str, default='ResNet18',
                   help='Model type ')
parser.add_argument('--train-split', type=float, default=0.8,
                   help='Fraction of training data to use for training (remainder used for validation)')
parser.add_argument('--data-augmentation', type=str, default='False',
                   help='Enable data augmentation for validation')

args = parser.parse_args()

RANDOM_SEED = 12345678
# Initialize device and seed
device = get_device()
set_seed(RANDOM_SEED)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get transforms and load datasets
transform_train, transform_test = get_transforms(args.dataset,args.data_augmentation)
trainset, valset, testset = load_datasets(args.dataset, transform_train,transform_test, train_split=args.train_split)



# Create data loaders

train_loader, val_loader, test_loader = create_dataloaders(
    trainset, valset, testset, args.batch_size, args.batch_size_val
)



# Create model

model = create_model(args.model_type, device)

# Initialize optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
cce = nn.CrossEntropyLoss(reduction="none")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max= 200,
    eta_min=1e-6
)

# Initialize cumulant
lambdas = torch.arange(-0.2, 0.5, step=0.01, device=device)
onlinecumulant = OnlineCumulant(device, args.buffer_size, forgetting_factor=args.forgetting_factor)

# Initialize metrics
test_metrics = init_simple_metrics()
start_epoch=0

for epoch in range(start_epoch, start_epoch+args.epochs):
    time_ep = time.time()
    columns = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'time']
    model.train()
    aux_loss=0
    #loop = tqdm(train_loader,unit="batch")
    zipped_data = zip(train_loader, cycle(val_loader))
    for batch, ((inputs_train, target_train), (inputs_val, target_val)) in enumerate(zipped_data):
        
        size_train = len(inputs_train)
        #size_val = len(inputs_val)
          # Clear gradients
        optimizer.zero_grad()
        inputs = torch.cat([inputs_train, inputs_val], dim=0).to(device)
        target = torch.cat([target_train, target_val], dim=0).to(device)
        #Forward pass with fresh computation graph
        logits = model(inputs)
        total_loss = cce(logits, target)
        batch_loss = torch.mean(total_loss[:size_train])
        val_losses = total_loss[size_train:]

        # Update statistics
        aux_loss += batch_loss.item()  
        avg_loss = aux_loss / (batch+1)

        weights = torch.ones_like(val_losses).to(device)
        weights = (weights / weights.sum()).detach()
        clone_val_losses = val_losses.clone().to(device)
        onlinecumulant.update_losses(clone_val_losses)
        rate, lambda_star = onlinecumulant.compute_rate_function(
                onlinecumulant.compute_mean() - avg_loss, 
                return_lambdas=True
        )
        rate = rate.clone().to(device) #constant gradient wrt model parameters.
        lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.01) #constant gradient wrt model parameters.
        model.train()
        cumulant = torch.logsumexp(-lambda_star * val_losses[:len(val_losses)//2] +\
        torch.log(weights[:len(val_losses)//2]), 0) + \
        torch.sum(weights * lambda_star * val_losses)
        inv_rate_with_grads = (rate + cumulant) / lambda_star
        loss = batch_loss + inv_rate_with_grads

        # Backpropagation
        loss.backward()
        optimizer.step()
        
    time_ep = time.time() - time_ep
    postfix_dict = update_simple_metrics(test_metrics, model, train_loader, test_loader,epoch)
    save_metrics(test_metrics, args, prefix='reg_')
    
    # using tabulate library to print,https://github.com/timgaripov/swa/blob/master/train.py
    values = [epoch,postfix_dict['Train cce'],postfix_dict['acc_train'], 
            postfix_dict['L_test'],postfix_dict['acc_test'],time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 0 :
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    scheduler.step()
