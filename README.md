# IKD Pre-Thesis: Sentiment Analysis & Baseline Modeling

This repository contains the pre-thesis research on sentiment analysis for the **Identitas Kependudukan Digital (IKD)** mobile application ([`gov.dukcapil.mobile_id`](https://play.google.com/store/apps/details?id=gov.dukcapil.mobile_id)) on Google Play Store.

The goal of this pre-thesis phase is to establish a rigorous data pipeline, evaluate preprocessing ablation techniques on informal Indonesian text, and build a strong baseline model using **TF-IDF + Support Vector Classifier (SVC)** with Bayesian optimization (**Optuna**) before proceeding to deep learning and Transformer-based models.

---

## 📌 Project Overview

* **Domain:** Indonesian Natural Language Processing (NLP) / Sentiment Analysis
* **Data Source:** Google Play Store reviews for the official IKD application
* **Classification Task:** Binary Sentiment Classification
  * `0`: Negative (Ratings 1–2)
  * `1`: Positive (Ratings 4–5)
  * *(Neutral reviews with rating 3 are filtered out)*
* **Baseline Architecture:** TF-IDF Vectorizer + Support Vector Machine (`SVC`)
* **Optimization:** Hyperparameter tuning via Optuna (with SQLite storage backend)

---

## 📂 Repository Structure

```text
├── 01_data_gathering.ipynb       # Scrapes reviews via google-play-scraper & uploads to Supabase
├── 02_data_cleaning.ipynb        # Cleans noise, filters rating 3, and assigns binary labels
├── 03_data_preprocessing.ipynb   # Slang normalization, cleaning, and stopword ablation variants
├── 04_baseline_model.ipynb       # TF-IDF + SVC baseline & preprocessing ablation analysis
├── 05_tune_baseline.ipynb        # Hyperparameter optimization using Optuna
├── Resource/
│   └── colloquial-indonesian-lexicon.csv  # Informal Indonesian dictionary for text normalization
├── Result/
│   ├── review_ikd_raw.csv                 # Raw scraped reviews
│   ├── ulasan_ikd_clean.csv               # Filtered & labeled reviews
│   └── ulasan_ikd_preprocessed.csv        # Preprocessed reviews with ablation columns
├── svm_tuning.sqlite3            # Optuna tuning study database
├── pyproject.toml                # uv project configuration and dependency declarations
├── uv.lock                       # Locked dependency tree for reproducibility
└── .python-version               # Pinned Python version (3.11)
```

---

## 🔬 Pipeline & Experimental Flow

1. **`01_data_gathering.ipynb` (Data Collection):**
   * Scrapes complete user reviews using `google-play-scraper`.
   * Persists raw data locally to `Result/review_ikd_raw.csv` and syncs with Supabase Storage.

2. **`02_data_cleaning.ipynb` (Data Cleaning & Labeling):**
   * Discards ambiguous/neutral reviews (Rating 3).
   * Maps ratings 1–2 to negative (`0`) and ratings 4–5 to positive (`1`).
   * Eliminates empty entries and duplicates, outputting `ulasan_ikd_clean.csv`.

3. **`03_data_preprocessing.ipynb` (Preprocessing & Ablation Setup):**
   * Text normalization using the Indonesian colloquial lexicon.
   * Cleans emojis, URLs, mentions, and special characters.
   * Generates three ablation text feature variants:
     * `text_baseline`: Cleaned & normalized text without stopword removal.
     * `text_standard`: Standard stopword removal using Sastrawi.
     * `text_custom`: Custom stopword removal preserving negation and domain-relevant terms.

4. **`04_baseline_model.ipynb` (Baseline Evaluation & Ablation Study):**
   * Compares the three preprocessing feature sets with default TF-IDF + Linear/RBF SVC.
   * **Key Finding:** `text_baseline` (retaining stopwords) yielded the highest Macro F1-Score (~0.9269), indicating that TF-IDF inverse document frequency effectively manages common word noise while preserving critical context and negation in short review texts.

5. **`05_tune_baseline.ipynb` (Hyperparameter Tuning):**
   * Splits data into Stratified Train (64%), Validation (16%), and Blind Test (20%).
   * Uses Optuna with TPE sampler to search over TF-IDF parameters (`ngram_range`, `min_df`, `max_df`) and SVM hyperparameters (`C`, `kernel`, `gamma`).
   * **Final Test Evaluation (Blind Test Set):**
     * **Macro F1:** ~0.9256
     * **Accuracy:** ~92.90%

---

## 🛠️ Environment Setup & Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python environment and dependency management.

### Prerequisites

Ensure you have `uv` installed on your machine:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd IKD-PreThesis
   ```

2. Synchronize dependencies:
   ```bash
   uv sync
   ```
   `uv` will automatically download and set up Python 3.11 along with all locked dependencies in `.venv/`.

### Configuration (`.env`)

For notebooks interacting with cloud storage (`01_data_gathering.ipynb`, `02_data_cleaning.ipynb`, etc.), create a `.env` file in the root directory:

```env
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-supabase-key"
SUPABASE_BUCKET="your-bucket-name"
```

---

## 🚀 Running the Notebooks

### Option 1: In VS Code / Cursor / PyCharm
Open any `.ipynb` file and select the virtual environment kernel:
```text
.venv/bin/python (macOS/Linux)
.venv\Scripts\python.exe (Windows)
```

### Option 2: Using JupyterLab via CLI
```bash
uv run jupyter lab
```

### Option 3: Inspecting Optuna Tuning Studies
To launch the Optuna Dashboard and visualize tuning trials:
```bash
uv run optuna-dashboard sqlite:///svm_tuning.sqlite3
```
