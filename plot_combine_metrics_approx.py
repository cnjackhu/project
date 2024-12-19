import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the data
def load_and_analyze_metrics():
    # Read CSV file
    df = pd.read_csv('data/combined_metrics.csv')
    
    # Filter for augmentation = False
    df = df[df['regularized'] == False]
    df = df[df['augmentation'] == False]
    df = df[df['train_split'] != 0.9999]


    #df = df[df['batch_size_val'] == 1000]

    # Create a configuration identifier column
    df['config'] = df.apply(lambda row: f"b{row['batch_size']}_v{row['batch_size_val']}_buf{row['buffer_size']}_"
                           f"ff{row['forgetting_factor']}_split{row['train_split']}_"
                           f"aug{row['augmentation']}_reg{row['regularized']}", axis=1)
    
    # Plot L_test evolution for each unique configuration
    plt.figure(figsize=(15, 10))
    
    # Get unique configurations
    configs = df['config'].unique()
    
    # Create a color palette for different configurations
    colors = sns.color_palette("husl", len(configs))
    
    metric_val = 'L_val'
    metric_test = 'L_test'
    metric_val = 'alphaD_val'
    metric_test = 'alphaD_test'
    metric_val = 'cummulants_val'
    metric_test = 'cummulants_test'
    #metric_val = 'var_val'
    #metric_test = 'var_test'

    # Plot each configuration
    for idx, config in enumerate(configs):
        config_data = df[df['config'] == config]
        # Add tiny random jitter to help distinguish overlapping lines
        jitter = np.random.normal(0, 0.001, size=len(config_data['L_test']))
      
        plt.plot(config_data['it'], config_data[metric_val], 
                label=config, color=colors[idx], alpha=0.7,
                linestyle='-')
    
        plt.plot(config_data['it'], config_data[metric_test] + jitter, 
                label=config, color=colors[idx], alpha=0.7,
                linestyle='--')

    plt.xlabel('Iterations')
    plt.ylabel(metric_test)
    plt.title(f'Evolution of {metric_test} Across Different Configurations')
    plt.grid(True, alpha=0.3)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('./data/l_test_evolution.png', bbox_inches='tight', dpi=300)
    plt.show()    

if __name__ == "__main__":
    load_and_analyze_metrics()