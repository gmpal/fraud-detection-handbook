# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter Book project documenting "Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook". The book is structured as a collection of Markdown files and Jupyter notebooks covering machine learning techniques for fraud detection.

**Online version**: https://fraud-detection-handbook.github.io/fraud-detection-handbook/

## Building and Development

### Build the Book

```bash
jupyter-book build .
```

The compiled book will be available at `_build/html/index.html`.

### Package Versions

The project was tested with:
- sphinxcontrib-bibtex==2.2.1
- Sphinx==4.2.0
- jupyter-book==0.11.2

See `requirements.txt` for full dependencies.

### Install Dependencies

```bash
pip install -r requirements.txt
```

For Binder environment (lightweight dependencies):
```bash
conda env create -f binder/environment.yml
```

## Repository Structure

### Chapter Organization

Content is organized in chapter directories:
- `Chapter_1_BookContent/` - Book overview and how to use
- `Chapter_2_Background/` - Credit card fraud background and ML approaches
- `Chapter_3_GettingStarted/` - Simulated dataset and baseline models
- `Chapter_4_PerformanceMetrics/` - Threshold-based, threshold-free, and top-k metrics
- `Chapter_5_ModelValidationAndSelection/` - Validation strategies and model selection
- `Chapter_6_ImbalancedLearning/` - Cost-sensitive learning, resampling, ensembling
- `Chapter_7_DeepLearning/` - Neural networks, autoencoders, sequential models
- `Chapter_References/` - Bibliography and shared functions

### Key Files

- `_config.yml` - Jupyter Book configuration
- `_toc.yml` - Table of contents structure
- `references.bib` - BibTeX references
- `Chapter_References/shared_functions.ipynb` - Reusable functions for data loading, model training, and performance assessment

### Shared Functions Module

The `Chapter_References/shared_functions.ipynb` notebook contains commonly reused functions throughout the book. It can be imported in other notebooks with:

```python
%run Chapter_References/shared_functions
```

Key function categories:
- **Data loading**: `read_from_files()` - Load pickle files from a date range
- **Train/test splitting**: `get_train_test_set()`, `prequentialSplit()` - Time-aware splitting with delay periods
- **Model fitting**: `fit_model_and_get_predictions()` - Train classifier and generate predictions
- **Performance assessment**: `performance_assessment()`, `card_precision_top_k()` - Evaluate fraud detection metrics
- **Model selection**: `prequential_grid_search()`, `model_selection_wrapper()` - Grid search with prequential validation
- **Deep learning utilities**: `FraudDataset`, `training_loop()`, `EarlyStopping` - PyTorch dataset and training helpers
- **Plotting**: `get_performances_plots()`, `plot_decision_boundary()` - Visualization functions

## Fraud Detection Domain Concepts

### Simulated Dataset

The book uses a custom transaction simulator (Chapter 3) that generates:
- Customer profiles with geographical locations, spending patterns
- Terminal profiles with locations
- Transactions with temporal features
- Three fraud scenarios:
  1. Amount > 220 (simple baseline)
  2. Compromised terminals (28-day periods)
  3. Compromised cards (14-day periods, amounts multiplied by 5)

Dataset characteristics:
- ~0.8% fraud rate (highly imbalanced)
- Mix of numerical and categorical features
- Time-dependent fraud scenarios
- Stored as daily pickle files in `simulated-data-raw/`

### Performance Metrics

The book emphasizes fraud detection-specific metrics:

**Card Precision@k (CP@k)**: The most important metric. Measures precision of detecting compromised cards in top-k most suspicious predictions per day. Removes detected cards from subsequent days.

**Prequential Validation**: Time-aware cross-validation that respects temporal ordering. Uses training period, delay period (simulates investigation time), and test period.

### Key Features

Transactions have core features:
- `TRANSACTION_ID`, `TX_DATETIME`, `CUSTOMER_ID`, `TERMINAL_ID`, `TX_AMOUNT`, `TX_FRAUD`
- `TX_TIME_SECONDS`, `TX_TIME_DAYS` - Temporal features

Feature engineering adds:
- Customer spending aggregates (RFM features)
- Terminal risk scores
- Time-based features

## Code Conventions

### Notebook Execution

- Notebooks are **not** executed during book build (`execute_notebooks: "off"` in `_config.yml`)
- Run notebooks locally or via Colab/Binder for interactive execution

### Data Loading Pattern

Data is split into daily pickle files. Load specific periods with:

```python
from Chapter_References.shared_functions import read_from_files

transactions_df = read_from_files(
    DIR_INPUT="./simulated-data-raw/",
    BEGIN_DATE="2018-04-01",
    END_DATE="2018-04-30"
)
```

### Model Training Pattern

Standard workflow:
1. Scale features with `scaleData()`
2. Split data with `get_train_test_set()` (respects delay period)
3. Fit model with `fit_model_and_get_predictions()`
4. Assess with `performance_assessment()` or `card_precision_top_k()`

### Time-Aware Splitting

Always use delay periods when splitting train/test to simulate investigation time:
- `delta_train`: Training period length (typically 7 days)
- `delta_delay`: Delay/investigation period (typically 7 days)
- `delta_test`: Test period length (typically 7 days)

Detected compromised cards are removed from test sets.

## Imbalanced Learning Context

This book focuses on highly imbalanced classification:
- Frauds are <1% of transactions
- Standard accuracy is misleading
- Use CP@k, AUC ROC, Average Precision
- Techniques covered: cost-sensitive learning, resampling (SMOTE, Tomek), ensemble methods

## Deep Learning Architecture

PyTorch is used for deep learning chapters:
- Custom `FraudDataset` for data loading
- `SimpleFraudMLPWithDropout` and `FraudMLP` for classification
- `SimpleAutoencoder` for anomaly detection
- `Attention` module for sequential modeling
- Utilities: `seed_everything()`, `training_loop()`, `EarlyStopping`

## Citations

If modifications reference the book, use:

```
@book{leborgne2022fraud,
  title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
  author={Le Borgne, Yann-Aël and Siblini, Wissam and Lebichot, Bertrand and Bontempi, Gianluca},
  url={https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook},
  year={2022},
  publisher={Université Libre de Bruxelles}
}
```

## Licenses

- Code: GNU GPL v3.0
- Prose and pictures: CC BY-SA 4.0
