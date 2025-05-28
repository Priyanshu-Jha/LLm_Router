# README

## Project Overview

This project is designed to evaluate and preprocess model-generated responses, train a similarity model using BERT, and perform inference through an API-like interface. It primarily focuses on multi-model response analysis with soft labels and textual similarity assessment.

---

## File Descriptions

### 1. **`response_preprocessing.py`**

A Python script for:
- **Preprocessing model responses** from multiple LLMs.
- **Cleaning artifacts** such as serialization markers, escape characters, and newline normalization.
- **Detecting response types** (code, math, text) and treating each accordingly.
- Produces a cleaned output file: `preprocessed_dataset.csv`.

Expected Input: `routerbench_no_mcq_with_groundtruth.csv`  
Output: `preprocessed_dataset.csv`

---

### 2. **`Soft_Label_Balanced_Dataset.csv`**

A CSV dataset (presumably soft-labeled) used for training or evaluating a similarity model. Typical format:
- Prompt and response pairs
- Label columns with soft scores
- Possibly balanced across label distributions

---

### 3. **`Model_Train_and_api_call.ipynb`**

A Jupyter notebook that likely includes:
- **Training of a text similarity model**, likely based on BERT or another transformer.
- **Evaluation metrics and loss tracking**.
- **API-like inference interface** for testing or deployment.

---

### 4. **`BERTSIM.ipynb`**

Another notebook likely focusing on:
- **Semantic similarity tasks** using BERT.
- Exploratory experiments with response embeddings, distance measures, or soft label regression.

---

## Requirements

- Python 3.8+
- pandas
- regex
- scikit-learn
- PyTorch + HuggingFace Transformers
- Jupyter Notebook

---

## How to Use

### 1. **Preprocessing**
```bash
python response_preprocessing.py
```
This cleans and normalizes responses from a CSV file and saves the output as `preprocessed_dataset.csv`.

### 2. **Model Training and Evaluation**
Open and execute:
- `Model_Train_and_api_call.ipynb`
- `BERTSIM.ipynb`

Ensure the cleaned dataset (`preprocessed_dataset.csv` or `Soft_Label_Balanced_Dataset.csv`) is loaded for model training or testing.

---

## Notes

- Response types (code, math, text) are automatically detected and cleaned accordingly.
- Additional heuristics are applied to the top 8227 rows for formatting consistency.
- The system supports multiple model responses including GPT, Claude, LLaMA, Mixtral, etc.

model links: https://iiithydstudents-my.sharepoint.com/:u:/g/personal/abhishek_g_students_iiit_ac_in/EeA1VKzUP7dPnVBS7FVnN2kBijlvywsAGzDBW9LxXpjBFg?e=Iu5x1k
