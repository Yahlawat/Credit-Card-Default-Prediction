
# Credit Card Default Prediction

This project predicts credit card default payments using machine learning models, with a focus on tree-based ensemble methods. The analysis is based on a dataset of Taiwanese credit card clients and emphasizes model evaluation, interpretability, and performance comparison.

<img src="./Images/logo.png" width="300" alt="UFC Logo">

## Steps & Techniques Used

- **Data Preprocessing**: Cleaning, categorical encoding, and feature engineering (e.g., delay trends, utilization ratios)
- **Class Imbalance Handling**: Applied SMOTE to oversample the minority class (defaulters)
- **Model Training**: Built and tuned Logistic Regression, Random Forest, AdaBoost, XGBoost, LightGBM, and CatBoost models using pipelines and RandomizedSearchCV
- **Model Evaluation**: Measured performance using F1 score, ROC AUC, confusion matrices, and ROC curves
- **Feature Importance**: Compared top predictors across tree-based models

## Results & Insights

- **Best Overall Models**: Random Forest and XGBoost offered the best balance of F1 score and recall, significantly outperforming a random baseline.
- **Key Predictors**: Max payment delay, credit limit, utilization ratios, and payment trends were the most influential features.
- **Interpretation**: Demographics had lower standalone importance but contributed through interactions.
- **Conclusion**: Ensemble models are well-suited for credit risk prediction, especially under class imbalance.


## Project Structure

```
.
├── notebooks/
│   └── default_prediction.ipynb    # Main analysis notebook
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── Images                          # Project Logo

```

## Dataset Summary 

The dataset includes 30,000 records of credit card clients in Taiwan, covering demographic information, bill and payment history, and default status over a 6-month period. [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data)

### Key Feature Groups
- **Demographics**: Age, Sex, Education, Marriage
- **Payment Behavior**: PAY_0 to PAY_6 (monthly repayment status)
- **Financials**: BILL_AMT1–6, PAY_AMT1–6, LIMIT_BAL
- **Target**: default.payment.next.month (1 = default, 0 = no default)

## Setup Instructions

```bash
git clone https://github.com/Yahlawat/Credit-Card-Default-Prediction.git
cd credit-card-default-prediction

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```
