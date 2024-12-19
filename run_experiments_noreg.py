import itertools
import subprocess
import os
from datetime import datetime

# Define parameter configurations
configs = {
    #'batch_size':[128], #[128,256,512,1024],
    # 'lr':[0.1,0.01,0.001],     #[0.1,0.01,0.001],
     'train-split':[0.8,0.9999],

    
   


    #'n_iters': [2001],  Keep constant for now
    #'train_split': [0.8],  # Keep constant for now
    #'data_augmentation': ['False'],  # Keep constant for now
}

cmd = ["python", "sgd_noreg_epoch.py"]

# Create output directory for logs
timestamp = datetime.now().strftime('%m%d_%H%M')
output_dir = f'long_exp_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Generate all combinations of parameters
keys, values = zip(*configs.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run experiments
for i, config in enumerate(experiments, 1):
    # Skip CNN for CIFAR10 if it's not implemented
    # if config['dataset'] == 'cifar10' and config['model_type'] == 'cnn':
    #     continue
        
    print(f"\nRunning experiment {i}/{len(experiments)}")
    print("Configuration:", config)
    
    # Build command
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Create log file name
    log_file = os.path.join(
        output_dir,
        f"exp_noreg_{i}_{'_'.join(f'{k}_{v}' for k, v in config.items())}.log"
    )
    
    # Run the experiment and log output
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            
        process.wait()
    
    print(f"Experiment {i} completed. Log saved to {log_file}")

print("\nAll experiments completed!") 