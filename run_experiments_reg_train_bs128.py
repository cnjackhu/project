import itertools
import subprocess
import os
from datetime import datetime

# Define parameter configurations
configs = {
    'batch_size': [128],
    'reg': [3,4,5],
    'lr': [0.1,0.05],
    'lamb': [0.1,0.5,1.0],
    'train_split': [0.9999],
    'data_augmentation':[True]# 'False'
}

cmd = ["python", "sgd_rate_reg_epoch_new.py"]

# Create output directory for logs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'experiment_logs_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Generate all combinations of parameters
keys, values = zip(*configs.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run experiments
for i, config in enumerate(experiments, 1):
       
    print(f"\nRunning experiment {i}/{len(experiments)}")
    print("Configuration:", config)
    
    # Build command
    for key, value in config.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Create log file name
    log_file = os.path.join(
        output_dir,
        f"exp_{i}_{'_'.join(f'{k}_{v}' for k, v in config.items())}.log"
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