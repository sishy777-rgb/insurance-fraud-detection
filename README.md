# 🛡️ Insurance Claim Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Latest-green)
![ML](https://img.shields.io/badge/ML-RandomForest-orange)
![SQL](https://img.shields.io/badge/Database-SQLite-lightgrey)

An AI-powered insurance claim fraud detection system built with domain knowledge 
of Hartford's Property & Casualty insurance business.

## 🎯 Why This Project?
Insurance fraud costs US insurers **$40 billion annually**. This system uses 
machine learning to automatically flag suspicious claims for investigation.

## 📊 Model Performance
| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| ROC-AUC Score | 0.82 |
| Fraud Recall | 82% (after threshold tuning) |
| Dataset | 1000 real insurance claims, 40 features |

## ⚡ Features
- Real-time fraud prediction with risk levels (HIGH/MEDIUM/LOW)
- SQL database storing all claims (SQLite + SQLAlchemy)
- Interactive web dashboard to submit and analyze claims
- Threshold tuning for business-optimized fraud detection
- Key fraud insights from SQL queries

## 🔍 Key Insights from Data
- Fraudulent claims average **$60,302 vs $50,288** for legitimate (20% higher)
- **Single & Multi vehicle collisions** are most common in fraud cases
- **incident_severity** is the strongest fraud predictor at 15.6% importance
- New customers (low months_as_customer) show higher fraud rates

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- SQLite + SQLAlchemy
- Flask (Web Dashboard)
- Joblib (Model Persistence)
- Git

## 📁 Project Structure

├── app.py                  # Flask web application
├── fraud.ipynb             # Data exploration & model training
├── templates/
│   └── index.html          # Web dashboard UI
├── fraud_model.pkl         # Trained Random Forest model
├── feature_columns.pkl     # Feature column names
├── insurance_claims.db     # SQLite database
├── insurance_claims.csv    # Raw dataset
└── requirements.txt        # Dependencies

## 🚀 Run Locally
```bash
git clone https://github.com/sishy777-rgb/insurance-fraud-detection
cd insurance-fraud-detection
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000

## 💡 Technical Decisions
- **Threshold tuned to 0.3** — In insurance, missing fraud is costlier than 
  false alarms. Lower threshold makes model more sensitive to fraud.
- **class_weight='balanced'** — Handles imbalanced dataset (75% legitimate, 
  25% fraud) without oversampling.
- **SQL over CSV** — Real insurance systems use relational databases. 
  SQLAlchemy simulates production data storage.

## 🏢 Business Relevance
Built specifically aligned with Hartford's P&C insurance domain. 
Demonstrates understanding of real fraud patterns and business tradeoffs 
in financial services.

