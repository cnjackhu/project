import torch


def compute_regularizer(lambda_star, val_losses, overlap=0.0):
    """
    Compute the regularizer with configurable overlap between sets.
    
    Args:
        lambda_star: The lambda parameter
        val_losses: Tensor of validation losses
        overlap: Float between 0 and 1, controlling the overlap between sets
                0.0 = no overlap (like compute_regularizer_1)
                1.0 = full overlap (like compute_regularizer_0)
    """
    n = val_losses.size(0)
    split_idx = int(n * (0.5 + overlap/2))  # Adjusts split point based on overlap
    
    first_set = val_losses[:split_idx]
    second_set = val_losses[n-split_idx:]  # Take from end to maintain set size
    
    partial_cumulant = torch.logsumexp(-lambda_star * first_set + torch.log(torch.tensor(1.0/split_idx, device=val_losses.device)), 0)
    
    return partial_cumulant/lambda_star +  second_set.mean(), second_set.mean(), partial_cumulant/lambda_star
