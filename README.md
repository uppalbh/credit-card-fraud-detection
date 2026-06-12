# Credit Card Fraud Detection Using Machine Learning

## Overview

Credit card fraud detection is a highly imbalanced classification problem where the objective is to accurately detect fraudulent transactions while minimizing false positives on legitimate transactions.

This project develops a machine learning pipeline to:
* Detect fraudulent transactions using classification models.
* Compare different sampling strategies for imbalanced datasets.
* Evaluate model performance using precision, recall, F1-score, and ROC-AUC.
* Analyze the impact of baseline vs undersampling vs SMOTE approaches.



## Business Problem

Fraudulent transactions are extremely rare but have very high financial impact. Standard machine learning models fail because accuracy becomes misleading in imbalanced datasets.

Early detection of fraud is critical for reducing financial losses and improving transaction security.

### Objectives
* Detect fraudulent transactions effectively.
* Handle extreme class imbalance properly.
* Compare multiple resampling techniques.
* Optimize models for recall (fraud detection priority).



## Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains anonymized credit card transaction data.

### Features

* Time – Seconds elapsed between transactions in dataset
* Amount – Transaction value
* V1 to V28 – PCA-transformed anonymized features

### Target Variable

* Class 0 = Legitimate transaction
* Class 1 = Fraudulent transaction



## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── ProjectBCreditCardFraud.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── models/
│   ├── baseline_model.pkl
│   ├── smote_model.pkl
│   └── undersample_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```



## Data Preprocessing

### Feature Scaling

* StandardScaler applied to:
  * Time
  * Amount



## Handling Class Imbalance

The dataset is highly imbalanced.

To address this issue:
* Baseline model trained on original data
* Undersampling applied to reduce majority class
* SMOTE used to generate synthetic minority samples



## Machine Learning Models Evaluated

All models use Logistic Regression for fair comparison.

### 1. Baseline Model
* Trained on original imbalanced dataset

### 2. Undersampled Model
* Balanced by reducing majority class

### 3. SMOTE Model
* Balanced using synthetic samples



## Model Evaluation

Metrics used:
* Accuracy (secondary)
* Precision
* Recall (primary)
* F1-score
* Classification report



## Results

### Class Distribution

Highly imbalanced dataset with very few fraud cases.

### Model Performance

| Model        | Precision | Recall | F1 Score |
|--------------|----------|--------|----------|
| Baseline     | 0.82     | 0.68   | 0.74     |
| SMOTE        | 0.14     | 0.91   | 0.24     |
| Undersampled | 0.03     | 0.92   | 0.06     |



## Visualizations

### EDA
* Class distribution plot
* Histograms + KDE
* Boxplots for outliers

### Model Evaluation
* Model comparison bar chart
* ROC Curve
* Precision-Recall tradeoff

### Confusion Matrix
* Baseline model CM
* SMOTE model CM



## Key Insights

* Fraud detection is recall-driven
* SMOTE improves fraud detection significantly
* Undersampling improves recall but reduces stability
* Imbalance handling matters more than model complexity



## Technologies Used

### Programming Language
* Python

### Libraries
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn



## Installation

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git

cd credit-card-fraud-detection

pip install -r requirements.txt
```



## Running the Project

```bash
jupyter notebook notebooks/ProjectBCreditCardFraud.ipynb
```

OR

```bash
python src/train.py
```



## Future Improvements

* XGBoost / LightGBM models
* Hyperparameter tuning
* FastAPI deployment
* Real-time fraud detection system
* Anomaly detection models
* Probability calibration



## Conclusion

This project shows how class imbalance handling techniques drastically affect fraud detection performance.

Recall-based evaluation is critical, and SMOTE provides the strongest improvement for detecting fraudulent transactions.
```
