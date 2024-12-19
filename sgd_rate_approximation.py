import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ratefunctiontorch import OnlineCumulant, RateCumulant
from utils import get_device, get_transforms, load_datasets, create_model, create_dataloaders, init_metrics, save_metrics, update_metrics

import torch
from torch import nn
from tqdm import tqdm
import argparse

# Keep existing argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fashion', choices=['cifar10', 'mnist', 'fashion'],
                   help='Dataset to use (cifar10 or mnist or fashion)')
parser.add_argument('--steps', type=int, default=50,
                   help='Number of steps between evaluations')
parser.add_argument('--batch-size', type=int, default=1000,
                   help='Training batch size')
parser.add_argument('--batch-size-val', type=int, default=250,
                   help='Validation/Test batch size')
parser.add_argument('--buffer-size', type=int, default=2000,
                   help='Buffer size for online cumulant (defaults to batch-size-val)')
parser.add_argument('--forgetting-factor', type=float, default=1.0,
                   help='Forgetting factor for online cumulant')
parser.add_argument('--n-iters', type=int, default=2001,
                   help='Number of training iterations')
parser.add_argument('--model-type', type=str, default='mlp', choices=['cnn', 'mlp'],
                   help='Model type for MNIST (cnn or mlp). Ignored for CIFAR10.')
parser.add_argument('--train-split', type=float, default=0.9999,
                   help='Fraction of training data to use for training (remainder used for validation)')
parser.add_argument('--data-augmentation', type=str, default='False',
                   help='Enable data augmentation for validation')

args = parser.parse_args()

RANDOM_SEED = 12345678
# Initialize device and seed
device = get_device()
torch.manual_seed(RANDOM_SEED)

# Get transforms and load datasets
transform, transform_val = get_transforms(args.dataset)
trainset, valset, testset = load_datasets(args.dataset, transform, train_split=args.train_split)

if args.data_augmentation == "True":
    valset.dataset.transform = transform_val

# Create data loaders
torch.manual_seed(RANDOM_SEED)
train_loader, val_loader, test_loader = create_dataloaders(
    trainset, valset, testset, args.batch_size, args.batch_size_val
)

if args.data_augmentation == "True":
    val_loader.dataset.transform = transform_val

# Create model
torch.manual_seed(RANDOM_SEED)
model = create_model(args.dataset, args.model_type, device)

# Initialize optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
cce = nn.CrossEntropyLoss(reduction="none")

# Initialize cumulant
lambdas = torch.arange(-0.2, 0.5, step=0.01, device=device)
onlinecumulant = OnlineCumulant(device, args.buffer_size, forgetting_factor=args.forgetting_factor)

# Initialize metrics
test_metrics = init_metrics()


torch.manual_seed(RANDOM_SEED)

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
    T_max=args.n_iters,
    eta_min=1e-6
)

torch.cuda.manual_seed(RANDOM_SEED)
tq = tqdm(range(args.n_iters))
for it in tq:
    
    model.train()

    # Get inputs and targets. If loader is exhausted, reinitialize.
    try:
        inputs_val, target_val = next(val_data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        val_data_iter = iter(val_loader)
        inputs_val, target_val = next(val_data_iter)

    # Move data to device
    inputs_val = inputs_val.to(device)
    target_val = target_val.to(device)

    onlinecumulant.update_losses_from_inputs(model, inputs_val, target_val)

    # Set model to train mode
    model.train()

    # Get inputs and targets. If loader is exhausted, reinitialize.
    try:
        inputs, target = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(train_loader)
        inputs, target = next(data_iter)

    # Move data to device
    inputs = inputs.to(device)
    target = target.to(device)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(inputs)

    # Compute the loss
    loss = torch.mean(cce(logits, target))
    
    # Update aux_loss and loss_count
    aux_loss += loss.detach().cpu().numpy()
    loss_count += 1  # Increment the count

    # Calculate the average loss
    avg_loss = aux_loss / loss_count


    # Backward pass
    loss.backward()
    # Update the weights
    optimizer.step()
    #scheduler.step()

    # Step the scheduler and check for early stopping
    if it % iters_per_epoch == 0 and it != 0:
        #if avg_loss < 0.01:
        #    break
        aux_loss = 0
        loss_count = 0  # Reset the count after each epoch
        
    if it % args.steps == 0 and it > 0:
        # Clear cache before heavy operations
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        postfix_dict = update_metrics(
            test_metrics, model, train_loader, test_loader,
            onlinecumulant, loss, optimizer, it
        )
        tq.set_postfix(postfix_dict)

        # After training loop, save metrics to CSV
        save_metrics(test_metrics, args,prefix='base')
