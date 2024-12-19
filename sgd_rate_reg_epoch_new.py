import sys
import os
import numpy as np
from itertools import cycle
from ratefunctiontorch import OnlineCumulant, RateCumulant
from utils import get_device,set_seed, get_transforms, load_datasets, create_model, \
create_dataloaders
import torch
from torch import nn
from tqdm import tqdm
import argparse
import time
from models import *
from metrics import init_metrics, save_metrics, update_metrics, get_argument_config, update_metrics_online
from regularizers import compute_regularizer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_config = get_argument_config()
for arg_name, arg_params in arg_config.items():
    parser.add_argument(
        f'--{arg_name}',
        f'--{arg_name.replace("_", "-")}',
        dest=arg_name,
        **arg_params
    )

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
val_onlinecumulant = OnlineCumulant(device, len(val_loader.dataset), loss_fn=cce)
train_onlinecumulant = OnlineCumulant(device, len(train_loader.dataset), loss_fn=cce)

# Initialize metrics
train_metrics = init_metrics()
val_metrics = init_metrics()
test_metrics = init_metrics()

tq = tqdm(range(args.epochs))
for epoch in tq:
    time_ep = time.time()
    model.train()
    aux_loss = 0
    aux_error = 0  # Initialize error counter
    zipped_data = zip(train_loader, cycle(val_loader))
    for batch, ((inputs_train, target_train), (inputs_val, target_val)) in enumerate(zipped_data):
        
        size_train = len(inputs_train)
        optimizer.zero_grad()
        inputs = torch.cat([inputs_train, inputs_val], dim=0).to(device)
        target = torch.cat([target_train, target_val], dim=0).to(device)
        
        model.train()

        logits = model(inputs)
        total_loss = cce(logits, target)
        train_losses = total_loss[:size_train]
        val_losses = total_loss[size_train:]

        val_onlinecumulant.update_losses(val_losses.clone().to(device))
        train_onlinecumulant.update_losses(train_losses.clone().to(device))


        batch_loss = torch.mean(train_losses)
        # Compute classification error (0-1 loss)
        predictions = torch.argmax(logits[:size_train], dim=1)
        batch_error = (predictions != target_train.to(device)).float().mean()
        
        # Update statistics
        aux_loss += batch_loss.item()
        aux_error += batch_error.item()  # Accumulate error
        train_loss = aux_loss / (batch+1)
        train_error = aux_error / (batch+1)  # Compute average error

        if args.reg <= 2:
            clone_val_losses = val_losses.clone().to(device)
            val_onlinecumulant.update_losses(clone_val_losses)
            rate, lambda_star = val_onlinecumulant.compute_rate_function(
                    val_onlinecumulant.compute_mean() - train_loss, 
                    return_lambdas=True
            )
            rate = rate.clone().to(device) #constant gradient wrt model parameters.
            lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.01) #constant gradient wrt model parameters.

        if args.reg == 0:
            regularizer, _, _ = compute_regularizer(lambda_star, val_losses, overlap=0.0)
        elif args.reg == 1:
            regularizer, _, _ = compute_regularizer(lambda_star, val_losses, overlap=1.0)
            loss = batch_loss + regularizer
        elif args.reg == 2:
            regularizer, _, _ = compute_regularizer(lambda_star, val_losses, overlap=0.5)
            loss = batch_loss + regularizer
        elif args.reg == 3:
            regularizer, batch_loss, _ = compute_regularizer(args.lamb, train_losses, overlap=0.0)
        elif args.reg == 4:
            regularizer, batch_loss, _ = compute_regularizer(args.lamb, train_losses, overlap=1.0)
        elif args.reg == 5:
            regularizer, batch_loss, _ = compute_regularizer(args.lamb, train_losses, overlap=0.5)

        if args.reg > 0:
            loss = batch_loss + regularizer
        else:
            loss = batch_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        
    time_ep = time.time() - time_ep
    
    update_metrics_online(train_metrics, train_onlinecumulant, train_loss)
    update_metrics_online(val_metrics, val_onlinecumulant, train_loss)
    update_metrics(test_metrics, model, test_loader, train_loss)
    
    save_metrics(train_metrics, args, folder_name='./data/full_metrics/', prefix='train_')
    save_metrics(val_metrics, args, folder_name='./data/full_metrics/', prefix='val_')
    save_metrics(test_metrics, args, folder_name='./data/full_metrics/', prefix='test_')
    
    postfix_dict = {
        'Train loss': train_loss,
        'Train acc': train_error,
        'Test loss': test_metrics['L'][-1],
        'Test acc': test_metrics['error'][-1]
    }
    tq.set_postfix(postfix_dict)

    scheduler.step()
