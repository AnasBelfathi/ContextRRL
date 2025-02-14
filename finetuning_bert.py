import os
import argparse
import pandas as pd
import numpy as np
import torch
import random

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch.distributed as dist


def set_seed(seed: int):
    """Fixe la seed pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data(csv_path, mini_test=False, split_name=""):
    """
    Lit un CSV et renvoie un DataFrame.
    Si mini_test=True, on ne garde que 300 premières lignes (pour debug).
    """
    print(f"\n--- Lecture du CSV {split_name}: {csv_path} ---")
    df = pd.read_csv(csv_path)
    print(f"-> Nombre de lignes initial: {len(df)}")

    if mini_test:
        df = df.head(300)
        print(f"-> [MINI TEST] On ne garde que {len(df)} lignes pour {split_name}.")

    return df


def tokenize_split(df, tokenizer, label_encoder, split_name="", model_name=""):
    """
    1) Encode la colonne 'target_label' du DataFrame via label_encoder.
    2) Tokenize la colonne 'all' avec (truncation=True, max_length=tokenizer.model_max_length).
    3) Retourne un tuple (df_mis_a_jour, dataset_hf).
    """
    df["encoded_label"] = label_encoder.transform(df["target_label"])

    print(f"\n[Tokenization] {split_name} - classes rencontrées : {df['target_label'].unique()}")
    if len(df) > 0:
        print(f"[Tokenization] {split_name} - Exemple d'encodage : {df['target_label'].iloc[0]} -> {df['encoded_label'].iloc[0]}")

    raw_texts = df["all"].tolist()
    # On détermine la longueur max depuis le tokenizer
    max_len = 1024 if model_name == "longformer-base-4096" else tokenizer.model_max_length

    print(f"[Tokenization] {split_name} - max_length={max_len}")

    tokenized = tokenizer(
        raw_texts,
        truncation=True,
        max_length=max_len
    )

    # Afficher un exemple de tokenisation, si le DF n'est pas vide
    if len(df) > 0:
        print(f"[Exemple {split_name}] premier texte : {raw_texts[0][:80]}...")
        tokens_example = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        print(f"[Exemple {split_name}] tokens : {tokens_example[:20]} ...")

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": df["encoded_label"].tolist()
    })

    return df, dataset


def compute_metrics(eval_pred):
    """Calcule accuracy et F1 macro."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def predict_and_save(trainer, df_input, dataset_input, label_encoder, output_csv_path):
    """
    Prédit sur dataset_input, ajoute 'predicted_label' au DataFrame df_input, sauvegarde en CSV.
    """
    preds_output = trainer.predict(dataset_input)
    preds = np.argmax(preds_output.predictions, axis=1)
    pred_labels_str = label_encoder.inverse_transform(preds)

    df_output = df_input.copy()
    df_output["predicted_label"] = pred_labels_str
    df_output.to_csv(output_csv_path, index=False)
    print(f"[Export] Fichier prédictions sauvegardé : {output_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--dev_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./finetune_outputs")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    parser.add_argument("--seed", type=int, default=42, help="Seed pour la reproductibilité.")
    parser.add_argument("--mini_test", action="store_true",
                        help="Si spécifié, on charge un petit subset (ex: 300 lignes) de chaque CSV.")
    parser.add_argument("--jeanzay", action="store_true",
                        help="Si vous tournez sur Jean Zay avec idr_torch (optionnel).")

    args = parser.parse_args()

    # Fix seed
    set_seed(args.seed)

    # Création du dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== Lancement du run (model={args.model_name}, seed={args.seed}) ===")

    # 1) Lecture des splits
    train_df = read_data(args.train_csv, mini_test=args.mini_test, split_name="train")
    dev_df   = read_data(args.dev_csv,   mini_test=args.mini_test, split_name="dev")
    test_df  = read_data(args.test_csv,  mini_test=args.mini_test, split_name="test")

    # 2) Labels globaux
    all_labels = pd.concat([train_df["target_label"], dev_df["target_label"], test_df["target_label"]], ignore_index=True)
    unique_labels = sorted(all_labels.unique())
    print(f"-> Ensemble complet de labels: {unique_labels}")
    label_encoder = LabelEncoder().fit(unique_labels)
    print(f"-> label_encoder.classes_ = {label_encoder.classes_}")

    # 3) Tokenizer & Model
    model_name = "./models/" + args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.model_name == "nomic-bert-2048":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(unique_labels),
            trust_remote_code = True,
            # attn_implementation="flash_attention_2" if model_name == "ModernBERT-base" else "sdpa"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(unique_labels),
            # attn_implementation="flash_attention_2" if model_name == "ModernBERT-base" else "sdpa"
        )

    # 4) Tokenisation
    train_df, train_dataset = tokenize_split(train_df, tokenizer, label_encoder, split_name="train", model_name= args.model_name)
    dev_df,   dev_dataset   = tokenize_split(dev_df,   tokenizer, label_encoder, split_name="dev", model_name= args.model_name)
    test_df,  test_dataset  = tokenize_split(test_df,  tokenizer, label_encoder, split_name="test", model_name= args.model_name)

    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        # load_best_model_at_end=True,
        # metric_for_best_model="f1_macro",  # ou "f1"
        # greater_is_better=True,
        save_total_limit=0,
        # bf16=True,
        # bf16_full_eval=True,
        # bf16=True, # bfloat16 training
        # optim="adamw_torch_fused",  # improved optimizer
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to="none"

    )

    # 6) Si on est sur Jean Zay (optionnel), initialiser le distributed
    if args.jeanzay:
        import idr_torch
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=idr_torch.size,
                                rank=idr_torch.rank)
        training_args.local_rank = idr_torch.local_rank

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=None,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # 8) Entraînement
    print("\n--- Début de l'entraînement ---")
    trainer.train()

    print("\n--- Évaluation sur dev ---")
    eval_metrics = trainer.evaluate(dev_dataset)
    print(f"Dev metrics: {eval_metrics}")

    # 9) Prédictions final sur dev / test
    dev_pred_csv  = os.path.join(args.output_dir, "dev_predictions.csv")
    test_pred_csv = os.path.join(args.output_dir, "test_predictions.csv")

    print("\n--- Prédictions sur dev & test ---")
    predict_and_save(trainer, dev_df, dev_dataset, label_encoder, dev_pred_csv)
    predict_and_save(trainer, test_df, test_dataset, label_encoder, test_pred_csv)

    print(f"=== Fin du run (model={args.model_name}, seed={args.seed}) ===")


if __name__ == "__main__":
    main()
