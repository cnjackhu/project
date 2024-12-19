import sys
import os
import numpy as np
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ratefunctiontorch import OnlineCumulant, RateCumulant
from utils import get_device,set_seed, get_transforms, load_datasets, create_model, \
create_dataloaders, init_metrics,init_simple_metrics, save_metrics, update_metrics,update_simple_metrics

import torch
from torch import nn
from tqdm import tqdm
import argparse
from models import *
# Keep existing argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['cifar10', 'mnist', 'fashion'], 
                    default='cifar10', help='Dataset to use: cifar10, mnist, or fashion')
parser.add_argument('--steps', type=int, default=50,
                   help='Number of steps between evaluations')
parser.add_argument('--batch-size', type=int, default=1000,
                   help='Training batch size')
parser.add_argument('--batch-size-val', type=int, default=125,
                   help='Validation/Test batch size')
parser.add_argument('--buffer-size', type=int, default=2000,
                   help='Buffer size for online cumulant (defaults to batch-size-val)')
parser.add_argument('--forgetting-factor', type=float, default=1.0,
                   help='Forgetting factor for online cumulant')
parser.add_argument('--n-iters', type=int, default=5000,
                   help='Number of training iterations')
parser.add_argument('--model-type', type=str, default='ResNet18',
                   help='Model type for MNIST (cnn or mlp). ')
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
cce = nn.CrossEntropyLoss(reduction="none")

# Initialize cumulant
lambdas = torch.arange(-0.2, 0.5, step=0.01, device=device)
onlinecumulant = OnlineCumulant(device, args.buffer_size, forgetting_factor=args.forgetting_factor)

# Initialize metrics
#test_metrics = init_metrics()
test_metrics = init_simple_metrics()
# Initialize data iterator
data_iter = iter(train_loader)

torch.manual_seed(RANDOM_SEED)
# Update the data iterator for validation
val_data_iter = iter(val_loader)

iters_per_epoch = len(data_iter)
aux_loss = 0
loss_count = 0  # Counter to keep track of the number of losses added

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max= int(args.n_iters/iters_per_epoch),
    eta_min=1e-6
)

torch.cuda.manual_seed(RANDOM_SEED)
tq = tqdm(range(args.n_iters))
for it in tq:
    
    model.train()

    # Validation phase
    try:
        inputs_val, target_val = next(val_data_iter)
    except StopIteration:
        val_data_iter = iter(val_loader)
        inputs_val, target_val = next(val_data_iter)

    try:
        inputs_train, target_train = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        inputs_train, target_train = next(data_iter)

    size_train = len(inputs_train)
    size_val = len(inputs_val)

    inputs = torch.cat([inputs_train, inputs_val], dim=0).to(device)
    target = torch.cat([target_train, target_val], dim=0).to(device)

    
    # Clear gradients
    optimizer.zero_grad()

    # Forward pass with fresh computation graph
    logits = model(inputs)
    total_loss = cce(logits, target)

    batch_loss = torch.mean(total_loss[:size_train])
    val_losses = total_loss[size_train:]

    # Update statistics
    with torch.no_grad():
        aux_loss += batch_loss.item()  
        loss_count += 1
        avg_loss = aux_loss / loss_count

    
    ###################################
    weights = torch.ones_like(val_losses).to(device)
    weights = (weights / weights.sum()).detach()
    
    clone_val_losses = val_losses.clone().to(device)
    clone_batch_loss = batch_loss.clone().to(device)
    
    onlinecumulant.update_losses(clone_val_losses)
    
    rate, lambda_star = onlinecumulant.compute_rate_function(
            onlinecumulant.compute_mean() - avg_loss, 
            return_lambdas=True
    )

    rate = rate.clone().to(device) #constant gradient wrt model parameters.
    lambda_star = torch.clamp(lambda_star.clone().to(device), min=0.01) #constant gradient wrt model parameters.
    
    model.train()
    if args.train_split < 0.9999:
        #cumulant = torch.logsumexp(-lambda_star * val_losses + torch.log(weights), 0) + torch.sum(weights * lambda_star * val_losses)
        #cumulant = torch.logsumexp(-lambda_star * val_losses + torch.log(weights), 0) + torch.sum(weights[:len(val_losses)//2] * lambda_star * val_losses[:len(val_losses)//2])
        cumulant = torch.logsumexp(-lambda_star * val_losses[:len(val_losses)//2] + torch.log(weights[:len(val_losses)//2]), 0) + torch.sum(weights * lambda_star * val_losses)

        inv_rate_with_grads = (rate + cumulant) / lambda_star

        # Compute total loss
        loss = batch_loss + inv_rate_with_grads
    else:
        loss = batch_loss


    loss.backward() 
    optimizer.step()
    
    # Step the scheduler and check for early stopping
    if it % iters_per_epoch == 0 and it != 0:
        #if avg_loss < 0.01:
        #    break
        aux_loss = 0
        loss_count = 0  # Reset the count after each epoch
        scheduler.step()
    if it % args.steps == 0 and it > 0:
        # Clear cache before heavy operations
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # postfix_dict = update_metrics(
        #     test_metrics, model, train_loader, test_loader,
        #     onlinecumulant, batch_loss, optimizer, it
        # )
        postfix_dict = update_simple_metrics(test_metrics, model, train_loader, test_loader,it)
        tq.set_postfix(postfix_dict)

        save_metrics(test_metrics, args, prefix='epoch_')

