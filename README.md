# ML_fraud_Citibank

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat-square)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4c72b0?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-ff6f00?style=flat-square)
![Imbalanced--Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11+-e74c3c?style=flat-square)
![Joblib](https://img.shields.io/badge/Joblib-1.3+-2ecc71?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Anaconda](https://img.shields.io/badge/Anaconda-Environment-44A833?style=flat-square&logo=anaconda)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-80.09%25-success?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.7224-blue?style=flat-square)

<p align="center">
  <img src="https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/header.png?raw=true" width="1000"/>
</p>

## Business Problem

Citibank launched the **Citi Gold Card**, a premium credit card targeting high-value customers. As the customer base grows, the bank faces a critical financial threat: **credit default**, which generates over **USD 50 billion in annual losses** across the financial industry.
Without a reliable predictive model, the bank cannot distinguish high-risk applicants from low-risk ones at the moment of credit approval, exposing the institution to avoidable financial damage. **Goal:** Build a Machine Learning classification model that predicts whether a customer will default (`Y = 1`) or not (`Y = 0`), based on their socioeconomic profile and monthly payment history.

## Dataset

| File | Description |
|---|---|
| `input/train1 (1).csv` | Training data — columns X1 to X7 + Y |
| `input/test_credit (1).csv` | Test data — columns X1 to X23 |
| `input/Sample_solution_credit (1).csv` | Submission sample |

### Feature Description

| Feature | Description |
|---|---|
| X1 | Credit limit (NT dollar) |
| X2 | Gender (1=Male, 2=Female) |
| X3 | Education (1=Graduate, 2=University, 3=High School, 0/4/5/6=Others) |
| X4 | Marital status (1=Married, 2=Single, 3=Divorce, 0=Others) |
| X5 | Age (years) |
| X6 | Repayment status — September 2005 |
| X7 | Repayment status — August 2005 |
| Y | Default (1=Yes, 0=No) — Target variable |

**Repayment status scale:** -2=No consumption, -1=Paid duly, 1=Delay 1 month ... 9=Delay 9+ months

## Project Structure

```
ML_fraud_Citibank/
│
├── img/                        # Charts and visualizations
├── input/                      # Raw dataset files
│   ├── train1 (1).csv
│   ├── test_credit (1).csv
│   └── Sample_solution_credit (1).csv
├── models/                     # Saved trained models (.pkl)
│   ├── gradient_boosting_optimized_calibrated.pkl
│   ├── gradient_boosting_baseline.pkl
│   ├── naive_bayes.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── adaboost.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   └── catboost.pkl
├── notebook/                   # Jupyter notebooks
│   └── citibank_fraud.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

## Pipeline

```
Raw Data
    │
    ▼
Exploratory Data Analysis (EDA)
    │   10 Business Questions
    │   Correlation Matrix
    │   Target Variable Analysis
    │
    ▼
Data Cleaning
    │   Null values check
    │   Duplicate removal
    │   Outlier treatment (IQR — X1, X5)
    │
    ▼
Class Balancing — SMOTE
    │   Before: 73.1% Non-Default / 26.9% Default
    │   After:  50.0% Non-Default / 50.0% Default
    │   Applied on training set only
    │
    ▼
Model Training — 9 Classifiers
    │   Naive Bayes, Decision Tree, Random Forest
    │   KNN, AdaBoost, Gradient Boosting
    │   XGBoost, LightGBM, CatBoost
    │
    ▼
Champion Model Optimization
    │   RandomizedSearchCV (50 iterations)
    │   StratifiedKFold (5 folds)
    │   CalibratedClassifierCV (isotonic)
    │
    ▼
Evaluation
    │   Confusion Matrix
    │   ROC Curve + AUC
    │   Classification Report
    │   Feature Importance + SHAP
    │
    ▼
Model Saving (.pkl)
```

## Exploratory Data Analysis

### Target Variable Distribution
![Target Variable](https://raw.githubusercontent.com/RafaelGallo/ML_fraud_Citibank/refs/heads/main/img/1.png)

The target variable `Y` shows a clear class imbalance. Out of 21,600 records, **16,766 customers (77.6%)** are Non-Default and **4,834 customers (22.4%)** are Default — a ratio of 3.5:1. Without treatment, models tend to be biased toward the majority class. This is why SMOTE was applied to synthetically balance the classes before model training.

| | Count | Share |
|---|---|---|
| Non-Default (0) | 16,766 | 77.6% |
| Default (1) | 4,834 | 22.4% |
| Imbalance Ratio | 3.5:1 | Non-Default : Default |

### Correlation Matrix
![Correlation Matrix](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/2.png?raw=true)

The correlation matrix reveals that **X6 and X7** (payment status in September and August) are the strongest predictors of default, with correlations of **0.32** and **0.26** respectively against target `Y`. X6 and X7 also show a strong correlation with each other (**0.67**), confirming that customers who delayed in one month tend to delay in the next. **X1** (credit limit) shows a moderate negative correlation with X6 and X7 (**-0.27** and **-0.29**), suggesting higher credit limits correlate with better payment behavior. Demographic variables (X2, X3, X4, X5) show very low correlations with Y, indicating limited individual predictive power.

| Variable | Correlation with Y | Signal |
|---|---|---|
| X6 — Sep payment | 0.32 | Strongest predictor |
| X7 — Aug payment | 0.26 | Second strongest |
| X6 × X7 | 0.67 | Consecutive delay pattern |
| X1 — Credit limit | -0.15 | Higher limit = lower risk |
| X2, X3, X4, X5 | < 0.05 | Weak individual signal |

### Credit Limit vs Default
![Credit Limit vs Default](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/3.png?raw=true)

Non-default customers have a significantly higher average credit limit of **NT$ 175,688** compared to **NT$ 128,984** for defaulters — a difference of **NT$ 46,704 (-26.6%)**. The boxplot confirms this pattern: non-defaulters show a wider interquartile range and higher median. Both groups present outliers reaching up to NT$ 1,000,000, suggesting that even high-limit customers can default, though less frequently. Higher credit limits tend to be granted to customers with better credit history and financial stability.

| | Non-Default | Default | Difference |
|---|---|---|---|
| Mean Credit Limit | NT$ 175,688 | NT$ 128,984 | -NT$ 46,704 (-26.6%) |
| Outliers | Up to NT$ 1,000,000 | Up to NT$ 750,000 | Lower ceiling for defaulters |

### Gender vs Default Rate
![Gender vs Default](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/4.png?raw=true)

**Male customers default at a higher rate (24.5%)** compared to **female customers (21.1%)** — a gap of 3.4 percentage points. The dataset contains significantly more female customers (~13,000) than male (~8,000), yet females maintain a lower default rate. The difference is relatively small and, as confirmed by the correlation matrix (X2 vs Y = -0.04), gender alone is not a strong predictor of default.

| | Default Rate | Total Customers |
|---|---|---|
| Female | 21.1% | ~13,000 |
| Male | 24.5% | ~8,000 |
| Difference | +3.4pp (Male higher) | — |

### Education Level vs Default Rate
![Education vs Default](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/5.png?raw=true)

Counterintuitively, **High School customers present the highest default rate at 25.1%**, followed by **University (23.8%)** and **Graduate (19.8%)**. The Others category shows only 7.0% but with negligible sample size. The spread between the three main groups is only 5.3 percentage points. As confirmed by the correlation matrix (X3 vs Y = 0.02), education has very weak individual predictive power and should be combined with payment history variables for meaningful risk signals.

| Education | Default Rate | Risk Ranking |
|---|---|---|
| High School | 25.1% | Highest |
| University | 23.8% | Second |
| Graduate | 19.8% | Third |
| Others | 7.0% | Lowest (small sample) |

### Age vs Default Rate
![Age vs Default](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/6.png?raw=true)

The age distribution is strongly right-skewed, with most customers concentrated between 20 and 40 years old. The **31-40 group presents the lowest default rate at 20.7%**, reflecting prime earning years and financial stability. Default rates increase progressively with age: 20-30 (22.8%), 41-50 (23.5%), 51-60 (24.9%), and 60+ (25.0%). Older customers may face fixed or declining incomes and higher healthcare costs. As confirmed by the correlation matrix (X5 vs Y = 0.01), age alone is a very weak predictor of default.

| Age Group | Default Rate | Risk Ranking |
|---|---|---|
| 31-40 | 20.7% | Lowest |
| 20-30 | 22.8% | Second lowest |
| 41-50 | 23.5% | Third |
| 51-60 | 24.9% | Second highest |
| 60+ | 25.0% | Highest |

### Payment Delay Correlation — Sep vs Aug
![Payment Delay Correlation](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/7.png?raw=true)

The heatmap and scatter plot confirm a strong behavioral consistency between September (X6) and August (X7) payment status, with a **Pearson correlation of 0.67**. The Paid duly × Paid duly cell dominates with 3,385 customers, followed by No consumption × No consumption with 1,746. Among delayed customers, Delay 1M × Delay 2M shows 1,223 occurrences, confirming a clear pattern of escalating or persistent delays. Defaulters are predominantly clustered in the positive quadrant (delays in both months), while non-defaulters concentrate in the negative region.

| Pattern | Count | Insight |
|---|---|---|
| Paid duly × Paid duly | 3,385 | Most stable customers |
| No consumption × No consumption | 1,746 | Inactive customers |
| Delay 1M × Delay 2M | 1,223 | Escalating delay pattern |
| X6 × X7 Correlation | 0.67 | Strong behavioral consistency |

## SMOTE — Class Balancing

![SMOTE Before vs After](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/10.png?raw=true)

SMOTE generated **7,453 synthetic samples** of the minority class (Default = 1), bringing both classes to perfect balance at **50/50**. The dataset grew from 16,139 to 23,592 samples. SMOTE was applied exclusively to the training set to prevent data leakage, ensuring the test set retained real-world class proportions for a fair evaluation.

| | Before SMOTE | After SMOTE |
|---|---|---|
| Non-Default (0) | 11,796 (73.1%) | 11,796 (50.0%) |
| Default (1) | 4,343 (26.9%) | 11,796 (50.0%) |
| Total | 16,139 | 23,592 |
| Synthetic samples | — | +7,453 |

## Machine Learning Models

Nine classification models were trained and evaluated using the SMOTE-balanced training set and evaluated on the original test set.

| Model | Type | Library |
|---|---|---|
| Naive Bayes | Probabilistic | Scikit-Learn |
| Decision Tree | Tree-based | Scikit-Learn |
| Random Forest | Ensemble | Scikit-Learn |
| KNN | Distance-based | Scikit-Learn |
| AdaBoost | Boosting | Scikit-Learn |
| Gradient Boosting | Boosting | Scikit-Learn |
| XGBoost | Boosting | XGBoost |
| LightGBM | Boosting | LightGBM |
| CatBoost | Boosting | CatBoost |

### Training Setup

| Item | Detail |
|---|---|
| **Train size** | 80% of dataset |
| **Test size** | 20% of dataset |
| **Stratify** | Yes — preserves class proportion |
| **SMOTE** | Applied on training set only |
| **Random state** | 42 — reproducibility |
| **Evaluation metric** | Accuracy (Kaggle competition metric) |

### Champion Model Optimization

The **Gradient Boosting** model was selected as champion and further optimized using the following techniques:

| Technique | Detail |
|---|---|
| **Search strategy** | RandomizedSearchCV — 50 iterations |
| **Cross-validation** | StratifiedKFold — 5 folds |
| **Calibration** | CalibratedClassifierCV — isotonic method |
| **Scoring** | Accuracy |
| **n_jobs** | -1 (all available cores) |

### Hyperparameter Grid

| Hyperparameter | Values Searched |
|---|---|
| n_estimators | 100, 200, 300, 500 |
| learning_rate | 0.01, 0.05, 0.1, 0.2 |
| max_depth | 3, 4, 5, 6 |
| min_samples_split | 2, 5, 10 |
| min_samples_leaf | 1, 2, 4 |
| subsample | 0.7, 0.8, 0.9, 1.0 |
| max_features | sqrt, log2, None |

## Model Evaluation

### Feature Importance — All Models
![Feature Importance](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/12.png?raw=true)

**X6 (September payment status)** emerges as the dominant predictor in Gradient Boosting (0.4989), XGBoost (0.3754) and CatBoost (26.94). **X7 (August payment status)** consistently ranks second in these models. Decision Tree and Random Forest rank **X5 (age)** first, followed by **X1 (credit limit)**, reflecting their reliance on demographic splits rather than behavioral signals. AdaBoost uniquely ranks **X3 (education)** as its top feature. LightGBM favors continuous variables X1 and X5 due to its split-gain based importance metric. Gender (X2) and marital status (X4) are the weakest predictors across all models.

| Rank | Feature | Description | Consensus |
|---|---|---|---|
| 1 | X6 | September payment status | Dominant in GB, XGBoost, CatBoost |
| 2 | X7 | August payment status | Strong in all boosting models |
| 3 | X1 | Credit limit | Dominant in DT, RF, LightGBM |
| 4 | X5 | Age | Strong in DT, RF, LightGBM |
| 5 | X3 | Education | Dominant in AdaBoost only |
| 6 | X4 | Marital status | Weak across all models |
| 7 | X2 | Gender | Weakest predictor in all models |

### Confusion Matrix — All Models

![Confusion Matrix All Models](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/13.png?raw=true)

Naive Bayes is the worst performer with 1,811 false positives — incorrectly flagging more than half of all non-defaulters as defaulters, making it commercially unviable. Decision Tree and Random Forest show moderate performance, catching fewer than half of actual defaulters. The boosting family (Gradient Boosting, XGBoost, LightGBM, CatBoost) achieves the best overall balance, with CatBoost recording the highest TN (2,665) and lowest FP (688) among all baseline models. AdaBoost achieves the highest TP (552) among the 72% accuracy models, making it the best at catching actual defaulters within that tier.

| Model | TN | FP | FN | TP | Accuracy |
|---|---|---|---|---|---|
| Naive Bayes | 1,542 | 1,811 | 280 | 687 | 52% |
| Decision Tree | 2,467 | 886 | 492 | 475 | 68% |
| Random Forest | 2,510 | 843 | 472 | 495 | 70% |
| KNN | 2,646 | 707 | 518 | 449 | 72% |
| AdaBoost | 2,578 | 775 | 415 | 552 | 72% |
| Gradient Boosting | 2,652 | 701 | 450 | 517 | 73% |
| XGBoost | 2,636 | 717 | 458 | 509 | 73% |
| LightGBM | 2,640 | 713 | 448 | 519 | 73% |
| CatBoost | 2,665 | 688 | 469 | 498 | 73% |

### ROC Curve — All Models

![ROC Curve All Models](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/14.png?raw=true)

AdaBoost achieves the highest AUC among baseline models (0.7135), followed closely by Gradient Boosting (0.7119) and LightGBM (0.7060). All boosting models fall in the acceptable AUC range (0.70–0.80) for a 7-feature dataset. Decision Tree has the lowest AUC (0.6201) due to its discrete, step-like decision boundaries producing poor probability calibration. Naive Bayes achieves a surprisingly decent AUC (0.6571) despite its worst accuracy, because AUC measures ranking ability rather than raw classification accuracy.

| Rank | Model | AUC | Classification |
|---|---|---|---|
| 1 | AdaBoost | 0.7135 | Acceptable |
| 2 | Gradient Boosting | 0.7119 | Acceptable |
| 3 | LightGBM | 0.7060 | Acceptable |
| 4 | CatBoost | 0.7012 | Acceptable |
| 5 | XGBoost | 0.6907 | Borderline |
| 6 | Random Forest | 0.6798 | Borderline |
| 7 | Naive Bayes | 0.6571 | Poor calibration |
| 8 | KNN | 0.6484 | Poor |
| 9 | Decision Tree | 0.6201 | Worst |

### Confusion Matrix — Champion Model

![Confusion Matrix Champion](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/15.png?raw=true)

The optimized model correctly classifies **3,065 true negatives** versus 2,848 in the baseline — an improvement of **217 additional correctly identified good customers (+7.6%)**. False positives drop dramatically from **505 to 288 (-43.0%)**, saving the bank from incorrectly flagging 217 good customers per 4,320 predictions. However, true positives fall from 512 to 395 (-22.9%) and false negatives increase from 455 to 572 (+25.7%). The optimization shifted the model's decision boundary toward higher specificity at the cost of lower sensitivity.

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| TN | 2,848 | 3,065 | +217 (+7.6%) |
| FP | 505 | 288 | -217 (-43.0%) |
| FN | 455 | 572 | +117 (+25.7%) |
| TP | 512 | 395 | -117 (-22.9%) |
| FPR | 15.06% | 8.59% | -6.47pp |
| FNR | 47.05% | 59.13% | +12.08pp |

### ROC Curve — Champion Model

![ROC Curve Champion](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/16.png?raw=true)

The optimized model's AUC of **0.7224** represents a meaningful improvement over the baseline's **0.7195** (+0.0029). The Gini coefficient improves from 0.4390 to 0.4448, confirming better overall discrimination between defaulters and non-defaulters. Both models fall in the acceptable AUC range (0.70–0.80), expected given the limited 7-feature dataset. The most significant curve improvement occurs in the critical low FPR region (0.15–0.40), where the optimized model maintains a more consistent upward trajectory due to isotonic calibration refining probability estimates.

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| AUC | 0.7195 | 0.7224 | +0.0029 |
| Gini Coefficient | 0.4390 | 0.4448 | +0.0058 |
| Industry Classification | Acceptable | Acceptable | — |

### Model Comparison — All Metrics

![Model Comparison](https://github.com/RafaelGallo/ML_fraud_Citibank/blob/main/img/17.png?raw=true)

The ranking table consolidates all 11 models by accuracy. Three distinct performance tiers are visible: the **boosting tier** (Gradient Boosting Optimized, GB Baseline, GB, CatBoost, LightGBM, XGBoost, AdaBoost) clustering between 72.45% and 80.09%, the **distance/tree tier** (KNN, Random Forest, Decision Tree) ranging from 69.56% to 71.64%, and **Naive Bayes** as a clear outlier at 51.60%. AdaBoost stands out as the model with the highest Recall (0.5708) and F1-Score (0.4813) among all baseline models, making it the best choice when maximizing defaulter detection is the priority over raw accuracy.

| # | Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|---|
| 1 | **GB Optimized + Calibrated** | **0.8009** | **0.5783** | 0.4085 | 0.4788 | **0.7224** |
| 2 | GB Baseline | 0.7778 | 0.5034 | 0.5295 | 0.5161 | 0.7195 |
| 3 | Gradient Boosting | 0.7336 | 0.4245 | 0.5346 | 0.4732 | 0.7119 |
| 4 | CatBoost | 0.7322 | 0.4199 | 0.5150 | 0.4626 | 0.7012 |
| 5 | LightGBM | 0.7312 | 0.4213 | 0.5367 | 0.4720 | 0.7060 |
| 6 | XGBoost | 0.7280 | 0.4152 | 0.5264 | 0.4642 | 0.6907 |
| 7 | AdaBoost | 0.7245 | 0.4160 | **0.5708** | **0.4813** | 0.7135 |
| 8 | KNN | 0.7164 | 0.3884 | 0.4643 | 0.4230 | 0.6484 |
| 9 | Random Forest | 0.6956 | 0.3700 | 0.5119 | 0.4295 | 0.6798 |
| 10 | Decision Tree | 0.6810 | 0.3490 | 0.4912 | 0.4081 | 0.6201 |
| 11 | Naive Bayes | 0.5160 | 0.2750 | 0.7104 | 0.3965 | 0.6571 |

## Final Conclusion

The **Gradient Boosting (Optimized + Calibrated)** model is the recommended submission for the Kaggle competition with **80.09% accuracy** and **AUC of 0.7224**. The single most actionable business insight from this pipeline is that **consecutive payment delays in two consecutive months multiply default probability by 3.4x** (57.1% vs 17.0%), making X6 and X7 the most valuable early warning signals for proactive credit risk management at Citibank. The classic Precision-Recall tradeoff was observed: the optimized model is more conservative and precise when flagging defaulters (Precision: 0.58 vs 0.50) but misses more actual defaulters (Recall: 0.41 vs 0.53). In a real banking scenario, if maximizing defaulter detection is the priority, AdaBoost (Recall: 0.5708) should be considered as an alternative.

### Business Recommendations

| Priority | Recommendation | Based On |
|---|---|---|
| Critical | Flag customers with consecutive delays (X6+X7 ≥ 1) as immediate high-risk | 57.1% default rate |
| Critical | Deploy Optimized model for Kaggle submission | 80.09% accuracy |
| High | Monitor X6 monthly as primary early warning indicator | Strongest single predictor |
| High | Apply lower credit limits to High School + young (20-30) segments | 25.1% + 22.8% default rates |
| Medium | Review credit limits for 60+ University customers proactively | 33.3% default rate |
| Medium | Use AdaBoost for high-risk portfolio screening requiring maximum recall | Highest recall 57.08% |
| Low | Collect additional features (income, debt ratio, tenure) | 7 features limit AUC ceiling at ~0.72 |

## Installation

```bash
# Clone the repository
git clone https://github.com/RafaelGallo/ML_fraud_Citibank.git
cd ML_fraud_Citibank

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
shap
joblib
```

## Usage

```python
import joblib
import pandas as pd

# Load champion model
model = joblib.load("models/gradient_boosting_optimized_calibrated.pkl")

# Load new data
df = pd.read_csv("input/train1 (1).csv")
X = df.drop(columns=["Ref.No", "Y"])

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

## Citation

```
Jayanth Rasamsetti. Citibank fraud defaulters.
https://kaggle.com/competitions/citibank-fraud-defaulters, 2018. Kaggle.
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Rafael Gallo**
[![GitHub](https://img.shields.io/badge/GitHub-RafaelGallo-black?style=flat-square&logo=github)](https://github.com/RafaelGallo)
