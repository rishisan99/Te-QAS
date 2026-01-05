# **TeQAS: Telugu Extractive Question Answering Dataset with Robust Span Alignment**

> **A large-scale Telugu extractive QA dataset constructed from SQuAD 2.0 with custom fuzzy span realignment (T-FuSAR) and extensive benchmarking using multilingual Transformer models.**

---

## 📌 Overview

Question Answering (QA) systems rely heavily on high-quality, span-aligned datasets. While English has mature QA resources (e.g., SQuAD 2.0), **Telugu remains a low-resource language**, limiting the effectiveness of modern Transformer-based QA models.

This project introduces **TeQAS**, a **large-scale Telugu extractive QA system** trained Telugu version SQuAD 2.0, addressing key challenges in multilingual QA:

* Accurate **answer span preservation after translation**
* Handling **morphologically rich Telugu structures**
* Robust support for **unanswerable questions**
* Benchmarking multilingual and Indic-specific models

In addition, we propose **T-FuSAR (Telugu Fuzzy Span Alignment & Realignment)** — a custom system that achieves **90.75% valid span retention with zero invalid spans**.

---

## 🚀 Key Contributions

* 📚 **TeQAS Dataset**

  * 130K+ Telugu QA pairs aligned with SQuAD 2.0 format
  * Supports both answerable and unanswerable questions
  * Compatible with HuggingFace QA pipelines

* 🧠 **T-FuSAR: Custom Span Realignment System**

  * Morphology-aware Telugu processing
  * Fuzzy character & token-level matching
  * Multi-occurrence answer resolution (94.8% success rate)

* 🔬 **Extensive Model Evaluation**

  * XLM-R Large (multilingual)
  * MuRIL Large (Indic-specialized)
  * Evaluation on TeQAS Test set, TeQuAD (Telugu SQuAD 1.0) + external benchmarks (TyDi QA, IndicQA, TeWiki QA)

---

## 📊 Dataset Details

### Source

* **Base Dataset:** SQuAD 2.0 (English)
* **Translation:** IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`)

### Dataset Splits

| Split      | Questions |
| ---------- | --------- |
| Train      | 109,680   |
| Validation | 8,595     |
| Test       | 10,846    |

* Preserves original SQuAD paragraph-wise structure
* Maintains compatibility with standard extractive QA models

---

## 🛠 Dataset Construction Pipeline

### 1️⃣ Translation

* Batched neural machine translation using IndicTrans2
* Robust error handling & resumable processing
* Semantic fidelity preserved across Telugu syntax variations

### 2️⃣ Answer Span Realignment — **T-FuSAR**

T-FuSAR resolves span misalignment caused by translation using:

**Morphological Analysis**

* Handles Telugu suffixes (లు, కి, తో, ని)
* Processes compound words and agglutination

**Semantic Grouping**

* Manages synonym variants and transliterated entities
* Preserves semantic equivalence

**Fuzzy Matching**

* Character & token-level similarity
* Adaptive window search
* Telugu Unicode boundary validation

📈 **Results**

* **118,275 / 130,319 spans preserved (90.75%)**
* **0 invalid spans**
* High robustness to word-order changes

---

### 3️⃣ Multi-Occurrence Resolution

When answers appear multiple times in context:

* Character-level indexing of all occurrences
* Parallel English–Telugu alignment
* Span selection based on English occurrence patterns

📌 **Resolution Rate:** 94.8% (2440 / 2575 cases)

---

## 🧪 Experimental Setup

### Models

* **XLM-R Large** – Cross-lingual Transformer (100+ languages)
* **MuRIL Large** – Indic-language specialized Transformer

### Preprocessing

* SentencePiece + WordPiece tokenization
* Telugu-specific normalization
* Noise & punctuation removal
* Span-safe text standardization

### Fine-Tuning

* Optimizer: AdamW
* Learning Rate: `2e-5`
* Epochs: 3 (Telugu)
* Max Sequence Length: 512
* Loss: Cross-Entropy (start/end span prediction)

### Metrics

* Exact Match (EM)
* F1 Score
* Is-Impossible Accuracy (unanswerable detection)

---

## 📈 Results

### 🔹 Full Dataset (Answerable + Unanswerable)

| Model | Setting                | Test EM   | Test F1   | Is_Im     |
| ----- | ---------------------- | --------- | --------- | --------- |
| XLM-R | Baseline (Pre-trained) | 0.20      | 0.22      | 39.31     |
| XLM-R | **Fine-tuned**         | **61.14** | **70.65** | **82.20** |
| MuRIL | Baseline (Pre-trained) | 0.24      | 0.26      | 47.34     |
| MuRIL | **Fine-tuned**         | 58.94     | 69.90     | 75.00     |

**Insights**

* Fine-tuning yields **massive gains (>60 EM / F1 points)** over baselines for both models.
* **XLM-R** demonstrates stronger **unanswerable question detection** (higher *Is_Im*).
* **MuRIL** remains competitive in span extraction despite lower unanswerable accuracy.

---

### 🔹 Answerable Questions Only

| Model | Setting                | Test EM   | Test F1   |
| ----- | ---------------------- | --------- | --------- |
| XLM-R | Baseline (Pre-trained) | 0.00      | 3.49      |
| XLM-R | **Fine-tuned**         | 51.23     | 71.66     |
| MuRIL | Baseline (Pre-trained) | 0.00      | 4.23      |
| MuRIL | **Fine-tuned**         | **51.60** | **72.09** |

**Insights**

* Baseline models fail to extract meaningful spans in Telugu without task-specific fine-tuning.
* **MuRIL consistently achieves higher F1**, indicating stronger alignment with Telugu morphology.
* Fine-tuning is **essential** for low-resource language QA performance.

---

### 🔹 Benchmark Evaluation

| Dataset          | XLM-R (EM / F1) | MuRIL (EM / F1)   |
| ---------------- | --------------- | ----------------- |
| TyDi QA (Telugu) | 55.46 / 71.94   | 53.66 / **77.09** |
| TeWiki QA        | 69.06 / 84.32   | **72.65 / 86.74** |
| IndicQA          | 53.23 / 70.82   | 53.66 / 71.00     |
| INDIC-Facts QA   | 67.29 / 84.69   | **68.03 / 84.57** |

**Insights**

* Best performance is observed on **Wikipedia-style datasets** (TeWiki QA).
* **MuRIL generalizes better** across Indic QA benchmarks.
* IndicQA remains challenging, highlighting long-context reasoning limitations.
* INDIC-Facts QA, a synthetic dataset developed during this project.

---

## 🧯 Error Analysis Highlights

* **Partial Span Errors:** Overly long predictions reduce EM
* **Entity Variations:** Morphological & suffix-based mismatches
* **Unanswerable Confusion:** False positives & negatives analyzed via confusion matrix

### Confusion Matrix Metrics

| Model | Accuracy   | Precision  | Recall     | F1         |
| ----- | ---------- | ---------- | ---------- | ---------- |
| XLM-R | 81.67%     | **0.8205** | 0.8114     | 0.8159     |
| MuRIL | **83.14%** | 0.7855     | **0.9123** | **0.8442** |

---

## 🔮 Future Work

* Expand dataset size and domain diversity
* Improve long-context reasoning
* Explore generative QA for Telugu
* Release pre-trained Telugu QA checkpoints

---

## 📦 Models & Dataset Availability

To support reproducibility and further research, both the **fine-tuned models** and the **TeQuAD dataset** are publicly released.

### 🤖 Fine-Tuned Models

Pretrained and fine-tuned QA models used in this work are available on Hugging Face:

* **Model Hub:**
  [https://huggingface.co/SanRish/Te-QAS-models/tree/main](https://huggingface.co/SanRish/Te-QAS-models/tree/main)

These include:

* XLM-R Large fine-tuned on TeQAS data (Telugu SQuAD 2.0)
* MuRIL Large fine-tuned on TeQAS data

---

### 📊 TeQuAD Dataset

The complete Telugu QA dataset is publicly available, including:

* Train / Validation / Test splits

* Original SQuAD 2.0–aligned structure

* **INDIC-Facts Telugu QA** synthetic factual dataset

* **Dataset Hub:**
  [https://huggingface.co/datasets/SanRish/Te-QAS/tree/main](https://huggingface.co/datasets/SanRish/Te-QAS/tree/main)

The dataset follows the standard **SQuAD 2.0 JSON format**, making it directly compatible with existing extractive QA pipelines and evaluation scripts.

---

### 🔁 Reproducibility

All models and datasets are released to enable:

* Reproducible evaluation
* Fair benchmarking on Telugu QA
* Extension to other Indic and low-resource languages

---


## 📜 License & Usage

* Intended for **research and academic use**
* Dataset structure follows SQuAD 2.0 format

---
