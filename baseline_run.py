import time
import gc
from datetime import datetime
from os import makedirs
import argparse
import torch
from eval_run import eval_and_save_metrics
from utils import get_device, ResultWriter, log
from task import pubmed_task
from train import SentenceClassificationTrainer
from models import BertHSLN
import os
import random
import numpy as np

# Import conditionnel pour Jean Zay
try:
    import idr_torch
    on_jean_zay = True
except ImportError:
    on_jean_zay = False

# Ajout des arguments
parser = argparse.ArgumentParser(description="Script de formation pour les modèles de classification de phrases.")
parser.add_argument("--task", type=str, required=True, help="Tâche à tester : category, rhetorical_function ou steps.")
parser.add_argument("--context_strategy", type=str, required=True, help="Stratégie de contexte (par ex. bm25, random, bertopic).")
parser.add_argument("--seed", type=int, required=True, help="Seed pour contrôler la reproductibilité.")
parser.add_argument("--tokenized_folder", type=str, required=True, help="Chemin vers le dossier contenant les fichiers tokenisés.")
parser.add_argument("--output_dir", type=str, required=True, help="Chemin vers le dossier contenant les fichiers tokenisés.")

args = parser.parse_args()

# Initialisation de la seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# BERT Model configuration
BERT_MODEL = "models/bert-base-uncased"

# Hyperparamètres du modèle
config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLN.__name__,
    "cacheable_tasks": [],
    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,
    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size": 32,
    "max_seq_length": 128,
    "max_epochs": 20,
    "early_stopping": 5,
}

MAX_DOCS = -1  # Vous pouvez ajuster ce paramètre si nécessaire.
# MAX_DOCS = 9
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS, data_folder=args.tokenized_folder, task_type=args.task)

task = create_task(pubmed_task)

# Configuration pour Jean Zay (si détecté)
if on_jean_zay:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )
    torch.cuda.set_device(idr_torch.local_rank)
    device = torch.device("cuda")
else:
    device = get_device(0)

# Définition des répertoires de sauvegarde des résultats
base_dir = f"{args.output_dir}/{args.task}/{args.context_strategy}/seed_{args.seed}"
makedirs(base_dir, exist_ok=True)

# Chargement des données
task.get_folds()

# Démarrage de l'entraînement
log(f"Début de l'entraînement pour la tâche : {args.task} avec la stratégie de contexte : {args.context_strategy} et seed : {args.seed}")

restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{base_dir}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} sur {task.num_folds}")
        result_writer.log(f"Début de l'entraînement pour le fold {fold_num}...")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=True, path=base_dir)
        if best_model is not None:
            model_path = os.path.join(base_dir, f"{restart}_{fold_num}_model.pt")
            result_writer.log(f"Sauvegarde du meilleur modèle dans {model_path}")
            torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"Fin de l'entraînement pour le fold {fold_num} en {time.time() - start:.2f} secondes.")
        gc.collect()

log("Entraînement terminé.")

# Calcul des métriques
log("Calcul des métriques...")
eval_and_save_metrics(base_dir, args.task, args.tokenized_folder)
log("Calcul des métriques terminé.")
