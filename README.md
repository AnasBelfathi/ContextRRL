# A Simple but Effective Context Retrieval for Sequential Sentence Classification in Long Legal Documents

<p align="center">
  <a href="https://aclanthology.org/2025.argmining-1.15/"><img src="https://img.shields.io/badge/Paper-ArgMining%202025-blue?style=flat-square&logo=read-the-docs" alt="Paper"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Workshop-ArgMining%20%40%20ACL%202025-orange?style=flat-square" alt="Workshop">
  <img src="https://img.shields.io/badge/Task-Sequential%20Sentence%20Classification-green?style=flat-square" alt="Task">
</p>

<p align="center">
  <b>Anas Belfathi &nbsp;·&nbsp; Nicolas Hernandez &nbsp;·&nbsp; Laura Monceaux &nbsp;·&nbsp; Richard Dufour</b><br>
  <i>Nantes Université, École Centrale Nantes, CNRS, LS2N, UMR 6004, F-44000, France</i>
</p>

---

## Abstract

Sequential sentence classification (SSC) assigns a functional label to each sentence based on its role within a document — a task that becomes especially challenging in long legal texts. State-of-the-art approaches face two major challenges: pre-trained language models (PLMs) struggle with input-length constraints, while hierarchical models often introduce noise from irrelevant content.

We propose a **simple and effective document-level retrieval approach** that extracts only the most relevant context for each target sentence. We introduce two families of heuristics:

- **Sequential** — captures local context based on positional proximity
- **Selective** — retrieves semantically similar sentences regardless of position

Experiments on three legal domain datasets show consistent improvements over the baseline, with an average increase of **~5.5 weighted-F1 points**. Sequential heuristics outperform hierarchical models on two out of three datasets, with gains of up to **~1.5 points**, demonstrating the benefit of targeted context over full-document processing.

---

## Context Retrieval Heuristics

### Sequential Heuristics
Extract sentences adjacent to the target sentence, preserving the natural document flow:

| Strategy | Description |
|---|---|
| `Before` | Selects the *k* sentences immediately **preceding** the target |
| `After` | Selects the *k* sentences immediately **following** the target |
| `Surrounding` | Selects *k/2* sentences **before and after** the target |

### Selective Heuristics
Retrieve sentences from anywhere in the document based on relevance:

| Strategy | Description |
|---|---|
| `Random` | Randomly selects *k* sentences from the document |
| `BM25` | Retrieves the *k* most relevant sentences via TF-IDF ranking |
| `Sentence-BERT` | Selects the *k* semantically closest sentences via siamese BERT embeddings |

Context length is fixed at **k = 6** for all experiments.

---

## Repository Structure

```
ContextRRL/
├── run_xps/                    # Experiment scripts
├── baseline_run.py             # BERT baseline without context
├── finetuning_bert.py          # Fine-tuning with context enrichment
├── global_context.py           # Selective heuristics (BM25, Sentence-BERT, Random)
├── local_context.py            # Sequential heuristics (Before, After, Surrounding)
├── models.py                   # Model architecture (BERT + BiLSTM + Attention)
├── tokenize_files.py           # Tokenization utilities
├── train.py                    # Training pipeline
└── README.md
```

---

## Model Architecture

The model builds on the hierarchical HSLN architecture with two modifications: the CRF layer is removed, and optimization focuses only on the target sentence enriched with retrieved context.

```
[Target Sentence + Retrieved Context]
        ↓
   BERT (Word Embedding)
        ↓
   Bi-LSTM + Attention Pooling  (Sentence Encoding)
        ↓
   Context Enrichment Layer     (Inter-sentence relationships)
        ↓
   Linear → Softmax             (Label prediction)
```

---

## Datasets

Experiments are conducted on three legal domain datasets, split at the document level (80/10/10):

| Dataset | Source | Sub-domain | Labels |
|---|---|---|---|
| DeepRhole | Bhattacharya et al. (2023) | Indian law | 7 classes |
| LegalEval | Kalamkar et al. (2022) | Indian law | 13 classes |
| SCOTUS | Lavissière & Bonnard (2024) | U.S. law | 13 classes |

---

## Results

Performance (Weighted F1) using the best context configuration at k ≤ 6. † and ‡ denote statistical significance over the baseline at p = 0.05 and p = 0.01.

### BERT (512 tokens) and Nomic-BERT (2048 tokens)

| Model | Heuristic | DeepRhole | LegalEval | SCOTUS |
|---|---|---|---|---|
| BERT | Baseline | 52.23 | 69.74 | 75.58 |
| | + Before | **67.18†** | 78.41† | 79.74† |
| | + After | 56.72† | **79.74†** | **81.34†** |
| | + Surrounding | 62.87† | 77.27† | 75.47 |
| | + BM25 | 51.59 | 69.43 | 75.96 |
| | + Sentence-BERT | 52.23 | 68.98 | 76.24 |
| Nomic-BERT | Baseline | 50.32 | 68.90 | 75.50 |
| | + Before | **67.89†** | 80.54† | 81.12† |
| | + After | 57.75† | **81.11†** | **81.32†** |
| | + Surrounding | 65.51† | 78.20† | 80.81† |
| | + BM25 | 53.90 | 70.82‡ | 77.06† |
| | + Sentence-BERT | 54.02‡ | 70.76‡ | 77.17‡ |
| BERT-HSLN | SOTA | 54.45 | **93.06** | 79.66 |

### Key Findings

- **Sequential heuristics consistently outperform selective ones** — positional proximity matters more than semantic similarity for SSC in legal texts.
- **`Before` is best on DeepRhole** (58.2%) — the dataset follows a progressive narrative where meaning builds on what came before.
- **`Surrounding` is best on LegalEval and SCOTUS** — rhetorical signals are distributed in both directions.
- **Selective heuristics yield marginal gains** — when documents lack semantically similar sentences, they add noise rather than signal.
- **Efficiency**: our retrieval-based models are **~3–5× lighter in GPU memory** and **~2–4× faster** in training/inference than BERT-HSLN.

---

## Installation

```bash
pip install transformers torch scikit-learn sentence-transformers rank_bm25
```

## Usage

### Run baseline (no context)

```bash
python baseline_run.py --dataset legaleval --model bert-base-uncased
```

### Run with Sequential context

```bash
python finetuning_bert.py \
    --dataset legaleval \
    --model bert-base-uncased \
    --heuristic before \
    --k 6
```

### Run with Selective context

```bash
python finetuning_bert.py \
    --dataset legaleval \
    --model bert-base-uncased \
    --heuristic bm25 \
    --k 6
```

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{belfathi-etal-2025-simple,
    title     = "A Simple but Effective Context Retrieval for Sequential Sentence 
                 Classification in Long Legal Documents",
    author    = "Belfathi, Anas and Hernandez, Nicolas and Monceaux, Laura and Dufour, Richard",
    booktitle = "Proceedings of the 12th Argument Mining Workshop",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.argmining-1.15/",
    pages     = "160--167"
}
```

---

## Acknowledgments

This research was funded in whole or in part by **l'Agence Nationale de la Recherche (ANR)**, project ANR-22-CE38-0004.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).