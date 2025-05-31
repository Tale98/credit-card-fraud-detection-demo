# credit-card-fraud-detection-demo
## Overview
- Dataset: Credit Card Fraud Detection (Kaggle) [Download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Problem: Binary classification – detect fraudulent credit card transactions among a highly imbalanced dataset.
- Techniques: Data analysis, feature engineering, imbalanced learning (SMOTE, RandomUnderSampler), cross-validation, model evaluation (AUC, F1), and SHAP-based feature importance.
## About the Dataset
  This project uses the popular Credit Card Fraud Detection dataset from Kaggle, which contains anonymized transactions made by European cardholders in September 2013.
  - Records: 284,807 transactions over two days
  - Fraudulent: 492 (0.17%) — Extremely imbalanced!
  - Features: Most are anonymized (V1–V28, from PCA), plus Time, Amount, and the target Class (0: legit, 1: fraud)
  - Goal: Build a classifier to accurately detect fraudulent transactions, despite strong class imbalance and anonymized features.
## Project Structure
  credit-card-fraud-detection-demo/
  - main.ipynb           # Jupyter notebook with all code & experiments
  - requirements.txt     # Python dependencies
  - README.md            # This file
  - creditcard.csv   # Raw dataset (NOT included, download separately)
## Quick Start
  1. Clone the repository
  ```bash
  git clone https://github.com/Tale98/credit-card-fraud-detection-demo
  ```
  2. Create venv
  ```bash
  python3 -m venv venv
  ```
  3. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
  4. Install Jupyter Notebook (in Visual Studio Code extension recommended)
     
     <img width="1498" alt="image" src="https://github.com/user-attachments/assets/c6ba43cd-e620-4585-a659-5b8ed2c7d69b" />
     
  5. Download the dataset : creditcard.csv from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  6. Run the notebook
     
     <img width="1458" alt="image" src="https://github.com/user-attachments/assets/dd9c2a66-a10c-49ed-9258-928785e3da54" />

# Features & Workflow
## Read Dataset as DataFrame

  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import plotly.io as pio
  from sklearn.model_selection import train_test_split
  pio.renderers.default = "notebook"
  df = pd.read_csv("creditcard.csv")
  ```
     
## Exploratory Data Analysis (EDA)

 ```python
  print(df.shape) ## check record number
  print(df.head()) ## explore example dataset records
  print(df.describe()) ## view summary dataset (counts, mean, std, etc)
  print(df.info())
  ```
  <img width="593" alt="image" src="https://github.com/user-attachments/assets/937dc217-7cd2-4f9a-aee3-cd369c0303e5" />

## Plot Class Distribution

  ```python
  plt.figure(figsize=(14, 6))
  df.groupby('Class').size().plot(kind='barh')
  ```

  The plot below shows the class distribution.It is immediately clear that the number of class 0 (legit) records is significantly higher than class 1 (fraud).In data science, this problem is known as an imbalanced class dataset.

  ![image](https://github.com/user-attachments/assets/d4030929-2074-4ee2-b6f3-ac656c81919c)

  Detecting fraud is challenging because the data is extremely imbalanced: there are only 492 fraudulent transactions out of more than 280,000 records. This makes standard metrics (like accuracy) misleading, and requires special care in model evaluation and training.

## Evaluation Metrics: Accuracy, Precision, and Recall
When working with imbalanced datasets (like credit card fraud detection), traditional metrics like accuracy can be misleading.
It is important to also look at precision and recall:
  1. Accuracy : The ratio of correct predictions to total predictions. (Example: If your model predicts all transactions as “not fraud” (class 0), it will have very high accuracy, but completely fail to detect fraud.)
  2. Precision : The ratio of true positives to all predicted positives. (all transactions the model flagged as fraud, how many were actually fraud?)
  3. Recall : The ratio of true positives to all actual positives. (Of all actual fraud transactions, how many did the model catch?)


- Why not just use Accuracy?

- Because fraud cases are so rare, a model could be 99.8% accurate by always predicting “not fraud”. That’s why precision and recall give a better sense of how well the model really works for the minority class.

## Imbalanced Data Handling Example
- Under Sampling : remove majority class
- Over Sampling : create new
     
