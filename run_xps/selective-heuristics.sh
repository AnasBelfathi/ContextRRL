#!/bin/bash

# Define directories
INPUT_ROOT="./ssc-datasets"
OUTPUT_ROOT="./processed-datasets-v2"
SCRIPT_PATH="./global_context.py"

# List of specific tasks to process (add the task names as seen in your dataset directory)
SELECTED_TASKS=("legal-eval" "scotus-rhetorical_function" "DeepRhole")

# Global context types to process
GLOBAL_CONTEXT_TYPES=("sentencebert")

# Context configuration
K_MIN=1
K_MAX=6

# Create output root if it doesn't exist
mkdir -p $OUTPUT_ROOT

# Function to show progress bar
show_progress_bar() {
    local completed=$1
    local total=$2
    local width=40  # Width of the progress bar
    local percent=$(( 100 * completed / total ))
    local progress=$(( width * completed / total ))
    local bar=$(printf "%-${width}s" "#" | cut -c1-$progress)

    printf "\r[%-${width}s] %d%% (%d/%d)" "$bar" "$percent" "$completed" "$total"
}

# Get the total number of tasks for progress calculation
TOTAL_TASKS=${#SELECTED_TASKS[@]}
CURRENT_TASK=0

echo "Starting global context data preparation..."

# Iterate over the selected tasks
for TASK_NAME in "${SELECTED_TASKS[@]}"; do
    TASK_DIR="$INPUT_ROOT/$TASK_NAME"
    CURRENT_TASK=$((CURRENT_TASK + 1))

    # Check if the directory exists and contains JSON files
    if [ ! -d "$TASK_DIR" ] || [ ! -f "$TASK_DIR/train.json" ]; then
        echo "No JSON files found for task: $TASK_NAME. Skipping..."
        show_progress_bar $CURRENT_TASK $TOTAL_TASKS
        continue
    fi

    # Set output directory for the task
    TASK_OUTPUT_DIR="$OUTPUT_ROOT/$TASK_NAME"
    mkdir -p "$TASK_OUTPUT_DIR"

    echo -e "\nProcessing task: $TASK_NAME ($CURRENT_TASK / $TOTAL_TASKS)"

    # Process each global context type
    CONTEXT_TOTAL=${#GLOBAL_CONTEXT_TYPES[@]}
    CONTEXT_COMPLETED=0

    for CONTEXT_TYPE in "${GLOBAL_CONTEXT_TYPES[@]}"; do
        CONTEXT_COMPLETED=$((CONTEXT_COMPLETED + 1))

        echo "  Processing global context type: $CONTEXT_TYPE ($CONTEXT_COMPLETED / $CONTEXT_TOTAL)"

        # Run the Python script for each split
        START_TIME=$(date +%s)
        python "$SCRIPT_PATH" --input_dir "$TASK_DIR" --output_dir "$TASK_OUTPUT_DIR" --k_min $K_MIN --k_max $K_MAX --context_type $CONTEXT_TYPE --randomize True
        END_TIME=$(date +%s)

        # Show time taken for each context type
        DURATION=$((END_TIME - START_TIME))
        echo "  Global context type $CONTEXT_TYPE processed in $DURATION seconds."

        # Update progress bar for the task
        show_progress_bar $CURRENT_TASK $TOTAL_TASKS
    done
done

echo -e "\nGlobal context data preparation complete!"
