#!/bin/bash

# Datasets to test
DATASETS=("mnist")

# Model types (only applicable for MNIST)
MODEL_TYPES=("mlp")

# Different batch sizes to test
BATCH_SIZE_VALS=(500 1000 2000)

# Different buffer sizes to test. Buffer size is the number of batches to store in the buffer.  
BUFFER_SIZES=(1 2 4)

# Different forgetting factors
FORGETTING_FACTORS=(0.95 0.99 1.0)

# Base configuration
STEPS=50
N_ITERS=2001
BATCH_SIZE=1000

# Create a log directory
mkdir -p logs

for DATASET in "${DATASETS[@]}"; do
    # For MNIST, test both model types
    if [ "$DATASET" == "mnist" ]; then
        MODEL_TYPES=("cnn" "mlp")
    else
        # For CIFAR10, only one model type
        MODEL_TYPES=("")
    fi

    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        for BATCH_SIZE_VAL in "${BATCH_SIZE_VALS[@]}"; do
            for BUFFER_SIZE in "${BUFFER_SIZES[@]}"; do
                BUFFER_SIZE = BUFFER_SIZE*BATCH_SIZE_VAL 
                for FF in "${FORGETTING_FACTORS[@]}"; do
                    # Create descriptive log filename
                    LOG_FILE="logs/run_${DATASET}_${MODEL_TYPE}_bs${BATCH_SIZE}_buf${BUFFER_SIZE}_ff${FF}.log"
                    
                    echo "Running experiment with:"
                    echo "Dataset: $DATASET"
                    echo "Model Type: $MODEL_TYPE"
                    echo "Batch Size Val: $BATCH_SIZE_VAL"
                    echo "Buffer Size: $BUFFER_SIZE"
                    echo "Forgetting Factor: $FF"
                    echo "Logging to: $LOG_FILE"
                    
                    # Construct the command
                    CMD="python demos/sgd_cumulant_approximation.py \
                        --dataset $DATASET \
                        --batch-size $BATCH_SIZE \
                        --batch-size-val $BATCH_SIZE_VAL \
                        --buffer-size $BUFFER_SIZE \
                        --forgetting-factor $FF \
                        --steps $STEPS \
                        --n-iters $N_ITERS"
                    
                    # Add model-type argument only for MNIST
                    if [ "$DATASET" == "mnist" ]; then
                        CMD="$CMD --model-type $MODEL_TYPE"
                    fi
                    
                    # Run the command and log output
                    $CMD > "$LOG_FILE" 2>&1
                    
                    echo "Finished experiment"
                    echo "----------------------------------------"
                done
            done
        done
    done
done

echo "All experiments completed!" 