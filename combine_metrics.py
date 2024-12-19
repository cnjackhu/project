import glob
import os
    
import pandas as pd
import re

def extract_config_from_filename(filename):
    """Extract configuration parameters from filename."""
    # Extract parameters using regex
    bs_match = re.search(r'bs(\d+)', filename)
    bsval_match = re.search(r'bsval(\d+)', filename)
    buf_match = re.search(r'buf(\d+)', filename)
    ff_match = re.search(r'ff([\d.]+)', filename)
    split_match = re.search(r'split([\d.]+)', filename)
    aug_match = re.search(r'aug(True|False)', filename)
    reg_match = re.search(r'(^|/)reg_', filename)
    mnist_match = re.search(r'mnist', filename.lower())
    fashion_match = re.search(r'fashion', filename.lower())
    mlp_match = re.search(r'mlp', filename.lower())
    cnn_match = re.search(r'cnn', filename.lower())
    
    return {
        'batch_size': int(bs_match.group(1)) if bs_match else None,
        'batch_size_val': int(bsval_match.group(1)) if bsval_match else None,
        'buffer_size': int(buf_match.group(1)) if buf_match else None,
        'forgetting_factor': float(ff_match.group(1)) if ff_match else None,
        'train_split': float(split_match.group(1)) if split_match else None,
        'augmentation': aug_match.group(1) == 'True' if aug_match else None,
        'regularized': bool(reg_match) if reg_match is not None else False,
        'mnist': bool(mnist_match) if mnist_match is not None else False,
        'fashion': bool(fashion_match) if fashion_match is not None else False,
        'mlp': bool(mlp_match) if mlp_match is not None else False,
        'cnn': bool(cnn_match) if cnn_match is not None else False
    }

def main():
    # Automatically find all relevant CSV files
    data_dir = './data'
    files = glob.glob(os.path.join(data_dir, 'reg_metrics_*.csv'))
    #files.extend(glob.glob(os.path.join(data_dir, 'reg_metrics_*.csv')))
    
    if not files:
        print("No matching CSV files found in ../data directory")
        return
    
    # Initialize empty list to store dataframes
    dfs = []
    
    # Process each file
    for file in files:
        # Read CSV
        df = pd.read_csv(file)
        
        # Extract configuration
        config = extract_config_from_filename(file.split('/')[-1])
        
        # Add configuration columns
        for key, value in config.items():
            df[key] = value
            
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataframe
    combined_df.to_csv('./data/combined_metrics.csv', index=False)
    print(f"Combined data saved to combined_metrics.csv")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    main() 