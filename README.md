# Credit Card Default Prediction

This project focuses on predicting credit card default payments using machine learning gradient boosting techniques. The analysis is performed on a dataset from Taiwanese credit card clients, comparing the performance of various models including Random Forest, AdaBoost, XGBoost, LightGBM, and CatBoost.

## Project Structure

```
.
├── notebooks/
│   └── default_prediction.ipynb    # Main analysis notebook
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Dataset Information

The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

### Features
- **Demographic Information**: ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
- **Payment History**: PAY_0 to PAY_6 (Repayment status from September to April 2005)
- **Bill Amounts**: BILL_AMT1 to BILL_AMT6 (Amount of bill statement)
- **Payment Amounts**: PAY_AMT1 to PAY_AMT6 (Amount of previous payment)
- **Target Variable**: default.payment.next.month (1=yes, 0=no)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Yahlawat/Credit-Card-Default-Prediction.git
cd credit-card-default-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/default_prediction.ipynb
```

2. Follow the notebook cells to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Train and evaluate different models
   - Analyze feature importance
   - Compare model performance

## Models Implemented

- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

## Results

The notebook includes comprehensive analysis of:
- Model performance metrics (accuracy, precision, recall, F1-score)
- ROC curves and AUC scores
- Feature importance analysis
- Model comparison and selection

## License

This project is licensed under the MIT License - see the LICENSE file for details. 