# ⚖️ ContextRRL — Simple but Effective Context Retrieval for Sequential Sentence Classification

[![Conference](https://img.shields.io/badge/ACL-2025-red.svg)](https://aclanthology.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ready-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

> 🔍 Official implementation of the paper **“A Simple but Effective Context Retrieval for Sequential Sentence Classification in Long Legal Documents”**,  
> accepted at **ACL 2025** 🏆  

---

## 🌟 Highlights

- 🚀 **Simple yet powerful** approach for context retrieval in Sequential Sentence Classification (SSC).  
- 🧩 Two complementary heuristics: **Sequential** (local) and **Selective** (semantic).  
- ⚡ **Improves PLMs by up to +5.5 F1** while being **3–5× more efficient** than hierarchical models.  
- 🧠 Demonstrates that *targeted context* beats full-document processing in long-text classification.  

---

## 🧠 Overview

**ContextRRL** investigates how to **retrieve the most relevant context** for sentence-level classification in long documents.  
Rather than relying on heavy hierarchical architectures, it introduces two simple yet highly effective retrieval strategies that enhance **pre-trained language models (PLMs)**:

- 🧭 **Sequential Context** — retrieves sentences before/after or surrounding the target.  
- 🔬 **Selective Context** — retrieves the most semantically similar sentences across the document.  

These heuristics make PLMs more efficient and context-aware for tasks such as **Rhetorical Role Labeling (RRL)** and **legal document understanding**.

---

## 📜 Abstract

Sequential sentence classification extends traditional classification to long documents where a sentence’s meaning depends on its surrounding context.  
However, **PLMs** struggle with input-length constraints, while **hierarchical models** often introduce irrelevant content.  

To address these issues, this paper introduces two **document-level retrieval heuristics**:
- **Sequential**, which captures *local positional information* (before, after, surrounding).  
- **Selective**, which retrieves *semantically similar sentences* independently of their position.  

Experiments on three legal datasets (**DeepRhole**, **LegalEval**, **SCOTUS**) show consistent improvements, with Sequential heuristics outperforming hierarchical models on two datasets and offering substantial efficiency gains.

---

## ⚙️ Architecture Overview

```mermaid
flowchart LR
  A[Target Sentence] --> B(Local Sequential Context)
  A --> C(Global Selective Context)
  B --> D[Fusion Layer]
  C --> D
  D --> E[Sentence Classifier]
  E --> F[Rhetorical Role Label]
