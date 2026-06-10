# Bank Loan Prediction System
### Predicting Loan Approvals and Maximum Loan Amounts using Machine Learning

---

## About This Project

This project applies supervised machine learning to real-world banking data to solve two problems:

- **Classification** — Can a client get a loan approved? *(Case Study A)*
- **Regression** — If approved, what is the maximum loan amount they qualify for? *(Case Study B)*

The work is split across three Jupyter notebooks, each handling a distinct phase of the machine learning pipeline.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `w2121365_Final_Python_Notebook_1.ipynb` | Exploratory data analysis, data cleaning, and dataset preparation |
| `w2121365_Final_Python_Notebook_2.ipynb` | Classification model building and hyperparameter optimisation |
| `w2121365_Final_python_notebook_3_.ipynb` | Ensemble methods and regression model development |

> **Run in order:** Notebook 1 → Notebook 2 → Notebook 3

---

## Files Required

Make sure the following CSV files are in the **same directory** as the notebooks before running:

```
loan_approval_data.csv          ← Original raw dataset (starting point)
Classification_Loan_Data.csv    ← Cleaned data for classification (output of Notebook 1)
Regression_Loan_Data.csv        ← Cleaned data for regression (output of Notebook 1)
```

> The raw dataset `loan_approval_data.csv` is included in this repository. The two prepared datasets are generated automatically when you run Notebook 1.

---

## Machine Learning Models

**Classification (Loan Approval Prediction)**
- Logistic Regression
- K-Nearest Neighbours — default and tuned via GridSearchCV
- Naïve Bayes
- Soft Voting Ensemble (Logistic Regression + Naïve Bayes)

**Regression (Maximum Loan Amount Prediction)**
- Decision Tree — fully grown
- Decision Tree — pruned with `max_depth=4`

---

## Results Summary

**Classification Performance**

| Model | F1 (Reject) | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.35 | 0.84 |
| KNN (k=5) | 0.48 | 0.81 |
| KNN Tuned (k=9, Manhattan) | 0.51 | 0.84 |
| Naïve Bayes | 0.44 | 0.81 |
| Soft Voting Ensemble | 0.43 | 0.83 |

**Regression Performance**

| Model | MSE | R² |
|---|---|---|
| Decision Tree (Full) | 447,895,126 | 0.85 |
| Decision Tree (Pruned, depth=4) | 829,592,976 | 0.73 |

---

## Tools & Libraries

- **Language:** Python 3
- **Data Handling:** Pandas, NumPy
- **Modelling:** Scikit-learn
- **Visualisation:** Plotly Express, Matplotlib

---

## Getting Started

1. Clone or download this repository
2. Open notebooks in **Google Colab** or a local **Jupyter** environment
3. Place all CSV files in the same folder as the notebooks
4. Execute notebooks sequentially starting from Notebook 1

---

*Author: Movindu Gamage | Module: 5DATA002W.2*
