#!/bin/bash
#SBATCH --job-name=finetune_bert  # Nom du job
#SBATCH --output=./job_out_err/%x_%A_%a.out  # Fichier de sortie
#SBATCH --error=./job_out_err/%x_%A_%a.err  # Fichier d'erreurs
#SBATCH -C v100-32g  # Type de GPU (ici V100 avec 32GB)
#SBATCH --ntasks=1  # Une seule tâche
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1  # Une seule GPU par tâche
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00  # Temps max d'exécution
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --array=0-71%72

set -e  # Arrêter en cas d'erreur

# -----------------------------------------------------------------------------
# Chargement des modules et activation de l'environnement
# -----------------------------------------------------------------------------
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.3.0


# -----------------------------------------------------------------------------
# Paramètres généraux
# -----------------------------------------------------------------------------
PROJECT_BASEPATH="."
SCRIPT_PATH="${PROJECT_BASEPATH}/finetuning_bert.py"
DATA_ROOT="${PROJECT_BASEPATH}/processed-datasets-modernBERT"
OUTPUT_ROOT="${PROJECT_BASEPATH}/finetune_results"
MINI_TEST=""  # Laissez vide ou mettez "--mini_test" pour un mini test
JEAN_ZAY="--jeanzay"

# Hyperparamètres
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5

# Listes de modèles, datasets, stratégies, seeds
MODELS=("bert-base-uncased")
DATASETS=("DeepRhole" "legal-eval" "scotus-rhetorical_function")
STRATEGIES=("left" "right" "neighbors" "none" "random" "bm25" "sentencebert")
SEEDS=("1" "2" "3")

# Calcul de l'indexation
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_STRATEGIES=${#STRATEGIES[@]}
NUM_SEEDS=${#SEEDS[@]}

TOTAL_JOBS=$((NUM_MODELS * NUM_DATASETS * NUM_STRATEGIES * NUM_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) dépasse le nombre total de jobs ($TOTAL_JOBS)."
    exit 1
fi

# Calcul des indices pour extraire le bon modèle, dataset, stratégie et seed
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_STRATEGIES * NUM_SEEDS)))
DATASET_INDEX=$(((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_SEEDS)) % NUM_DATASETS))
STRATEGY_INDEX=$(((SLURM_ARRAY_TASK_ID / NUM_SEEDS) % NUM_STRATEGIES))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

MODEL_NAME=${MODELS[$MODEL_INDEX]}
DATASET_NAME=${DATASETS[$DATASET_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# -----------------------------------------------------------------------------
# Définition des fichiers CSV pour l'entraînement et le test
# -----------------------------------------------------------------------------
TRAIN_CSV="${DATA_ROOT}/${DATASET_NAME}/${STRATEGY}/train.csv"
DEV_CSV="${DATA_ROOT}/${DATASET_NAME}/${STRATEGY}/dev.csv"
TEST_CSV="${DATA_ROOT}/${DATASET_NAME}/${STRATEGY}/test.csv"

# Vérification des fichiers
if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "Fichier introuvable: $TRAIN_CSV. On saute."
    exit 1
fi
if [[ ! -f "$DEV_CSV" || ! -f "$TEST_CSV" ]]; then
    echo "Fichiers dev/test manquants pour $DATASET_NAME/$STRATEGY. On saute."
    exit 1
fi

# -----------------------------------------------------------------------------
# Construction du chemin de sortie
# -----------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${DATASET_NAME}/${STRATEGY}/seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

echo -e "\n--- Lancement: MODEL=$MODEL_NAME, DATASET=$DATASET_NAME, STRATEGY=$STRATEGY, SEED=$SEED ---"

# -----------------------------------------------------------------------------
# Lancement du script Python
# -----------------------------------------------------------------------------
srun python "$SCRIPT_PATH" \
    --train_csv "$TRAIN_CSV" \
    --dev_csv "$DEV_CSV" \
    --test_csv "$TEST_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --seed "$SEED" \
    $MINI_TEST \
    $JEAN_ZAY

# Vérification d'erreur
if [ $? -ne 0 ]; then
    echo "Erreur lors du fine-tuning pour MODEL=$MODEL_NAME, DATASET=$DATASET_NAME, STRATEGY=$STRATEGY, SEED=$SEED"
    exit 1
fi

echo "Fin du fine-tuning pour MODEL=$MODEL_NAME, DATASET=$DATASET_NAME, STRATEGY=$STRATEGY, SEED=$SEED."
