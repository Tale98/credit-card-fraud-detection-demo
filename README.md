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
When the dataset is highly imbalanced (e.g., very few frauds vs. many legit transactions), we can use these strategies:
- Under Sampling : Randomly remove some records from the majority class (class 0: legit) until the dataset is more balanced.

  Pros: Simpler, faster to train.

  Cons: May lose important information from the majority class.
  
- Over Sampling : Create new synthetic records for the minority class (class 1: fraud) using techniques like SMOTE, or simply duplicate existing minority class examples until the classes are balanced.

  Pros: Keeps all original data, gives the model more fraud examples to learn from.

  Cons: Can lead to overfitting if not done carefully.
  
Commonly used libraries:
- Under sampling: imblearn.under_sampling.RandomUnderSampler
- Over sampling: imblearn.over_sampling.SMOTE

## Features Engineering Model Based

```python
  ## pipeline
  from imblearn.pipeline import Pipeline
  from sklearn.ensemble import RandomForestClassifier
  ## pipeline random over sampling
  pipeline_random = Pipeline([
      ("scaler", Scaler()),
      ('oversampling', RandomOverSampler(random_state=42)),
      ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
  ])
  ## pipeline smote
  pipeline_smote = Pipeline([
      ("scaler", Scaler()),
      ('oversampling', SMOTE(random_state=42)),
      ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
  ])
  pipelines = {
      "Random Under Sampling": pipeline_random,
      "SMOTE": pipeline_smote
  }
  for name, pipeline in pipelines.items():
      print(f"Training pipeline: {name}")
      # Fit the pipeline
      pipeline.fit(X_train, y_train)
      # Evaluate the model
      from sklearn.metrics import classification_report, confusion_matrix
      y_pred = pipeline.predict(X_test)
      print(f"Results for {name}:")
      print(classification_report(y_test, y_pred))
      print(confusion_matrix(y_test, y_pred))
      print("\n")
  ```

- Sampling Selection
  
<img width="433" alt="image" src="https://github.com/user-attachments/assets/3afbbf15-b158-4df4-9bf4-fdec9faab7d2" />

From result I prefered Training pipeline: SMOTE because better recall

```python
  ## feature importance
  import shap
  selected_pipeline = pipelines["SMOTE"]
  explainer = shap.TreeExplainer(selected_pipeline.named_steps["classifier"])
  
  class1_index = y_test[y_test == 1].index
  shap_values = explainer.shap_values(X_test.loc[class1_index])
```

I will focus on class 1 True Positive (fraudulent transactions correctly identified) to analyze which feature improve the prediction of fraud transactions.

```python
  ## feature importance plot
  shap.summary_plot(shap_values[:, :, 1], X_test.loc[class1_index], plot_type="bar")
```

![image](https://github.com/user-attachments/assets/1e4ce4e9-2672-4813-aae3-3b33fdc040bf)

The plot above show us Feature Importance then we can select base on top K features 

```python
  feature_importance = pd.Series(shap_values[:, :, 1].mean(axis=0), index=X_test.columns)
  feature_importance.sort_values(ascending=False, inplace=True)
  k = 10
  selected_features = feature_importance.head(k).index.tolist()
  print(f"Top {k} features based on SHAP values:")
  print(selected_features)
```
Top 10 features based on SHAP values:
['V14', 'V10', 'V17', 'V12', 'V4', 'V3', 'V11', 'V16', 'V2', 'V21']

## Train new model with selected features

```python
  ## train a new model with selected features
  X_train_selected = X_train[selected_features]
  X_test_selected = X_test[selected_features]
  pipeline_selected = Pipeline([
      ("scaler", Scaler()),
      ('oversampling', SMOTE(random_state=42)),
      ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
  ])
  pipeline_selected.fit(X_train_selected, y_train)
  y_pred_selected = pipeline_selected.predict(X_test_selected)
  from sklearn.metrics import classification_report, confusion_matrix
  print("Results for model with selected features:")
  print(classification_report(y_test, y_pred_selected))
  print(confusion_matrix(y_test, y_pred_selected))
```

<img width="405" alt="image" src="https://github.com/user-attachments/assets/40e9a871-1022-40da-8065-7c01f870ebe3" />

## Summary & Key Findings

This project tackles the Credit Card Fraud Detection problem using a highly imbalanced public dataset, where fraudulent (class 1) transactions account for less than 0.2% of all records.

What we did:
- Data exploration confirmed extreme class imbalance.
- Addressed the imbalance with SMOTE (Synthetic Minority Oversampling Technique) to upsample the minority (fraud) class for better learning.
- Built robust pipelines with scikit-learn and imblearn, comparing models with all features vs. only the Top 10 features selected by SHAP (feature importance/explainability).
- Evaluated models using metrics: accuracy, precision, recall, f1-score, focusing especially on recall for class 1 (fraud).

Key results:
- SMOTE + All features:
- Precision (fraud): 0.85
- Recall (fraud): 0.84
- F1-score (fraud): 0.85
- SMOTE + Top SHAP features:
- Precision (fraud): 0.80
- Recall (fraud): 0.84
- F1-score (fraud): 0.82
- Using SHAP-based feature selection helps reduce model complexity, with a small trade-off in precision but similar recall (which is crucial for fraud detection).

Interpretation:
- Both pipelines manage class imbalance well and yield high recall for fraud, which is the key metric in this domain.
- Feature selection (via SHAP) makes the model more interpretable and lighter, with only a small drop in precision.
- The workflow demonstrates how sampling, feature importance, and proper evaluation can build practical, explainable fraud detection models.
  
# References
- [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Imbalanced-learn documentation (imblearn)](https://imbalanced-learn.org/stable/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html)

# Contributors
- Apisit 
  - [GitHub](https://github.com/Tale98)
  - [LinkedIn](https://www.linkedin.com/in/apisit-chiamkhunthod-2a11221b4/)
  - Email: oh.oh.159852357@gmail.com

Feel free to reach out if you have any questions or want to collaborate!
