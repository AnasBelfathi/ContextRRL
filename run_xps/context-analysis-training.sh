#!/bin/bash
#SBATCH --job-name=bert_training
#SBATCH --output=./job_out_err/%x_%A_%a.out
#SBATCH --error=./job_out_err/%x_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --array=0-8%9  # Total jobs = #TASKS * #CONTEXT_STRATEGIES * #SEEDS

set -e

# Chargement des modules
module purge
module load cpuarch/amd
module load anaconda-py3/2024.06
conda activate my_new_env

# Définir les tâches, contextes et seeds
TASKS=("scotus-rhetorical_function")
CONTEXT_STRATEGIES=("bm25" "sentencebert" "entities")
#CONTEXT_STRATEGIES=("none")

SEEDS=(1 2 3)

# Base paths
BASE_TOKENIZED_FOLDER="processed-datasets-v2"
OUTPUT_FOLDER="output-training-v2"

# Calcul des indices pour les tableaux
NUM_CONTEXTS=${#CONTEXT_STRATEGIES[@]}
NUM_SEEDS=${#SEEDS[@]}

TASK_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_CONTEXTS * NUM_SEEDS)))
CONTEXT_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_SEEDS) % NUM_CONTEXTS ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

# Sélection des valeurs
TASK=${TASKS[$TASK_INDEX]}
CONTEXT=${CONTEXT_STRATEGIES[$CONTEXT_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# Définir le chemin du dossier tokenized
TOKENIZED_FOLDER="$BASE_TOKENIZED_FOLDER/$TASK/$CONTEXT"

# Vérifier l'existence du dossier tokenized
if [ ! -d "$TOKENIZED_FOLDER" ]; then
    echo "Error: Tokenized folder $TOKENIZED_FOLDER not found. Skipping..."
    exit 1
fi

# Créer le dossier de sortie si nécessaire
mkdir -p $OUTPUT_FOLDER

# Exécuter le script de formation
echo "============================="
echo "TASK: $TASK"
echo "CONTEXT: $CONTEXT"
echo "SEED: $SEED"
echo "============================="

srun python baseline_run.py \
    --task $TASK \
    --context_strategy $CONTEXT \
    --seed $SEED \
    --tokenized_folder $TOKENIZED_FOLDER \
    --output_dir $OUTPUT_FOLDER

echo "Training completed for TASK=$TASK, CONTEXT=$CONTEXT, SEED=$SEED."
