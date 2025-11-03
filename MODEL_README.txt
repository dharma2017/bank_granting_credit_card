
# Credit Card Default Prediction Model - Deployment Package

## Model Information
- **Model Name:** Random Forest (Tuned)
- **Model Type:** RandomForestClassifier
- **Training Date:** 2025-11-03 10:15:28
- **Test Accuracy:** 0.7767
- **F1 Score:** 0.5176
- **Precision:** 0.4955
- **Recall:** 0.5418

## Files Included
1. `best_credit_model.pkl` - Trained model
2. `scaler.pkl` - StandardScaler for feature preprocessing
3. `feature_names.pkl` - List of features in correct order
4. `feature_config.pkl` - Feature engineering configuration
5. `model_info.pkl` - Complete model metadata
6. `model_info.json` - Model metadata (human-readable)
7. `split_info.pkl` - Train/test split information
8. `all_models_comparison.csv` - All models performance comparison
9. `confusion_matrix.pkl` - Confusion matrix and metrics for best model

## Confusion Matrix Results
- True Negatives (TN): 3941
- False Positives (FP): 732
- False Negatives (FN): 608
- True Positives (TP): 719

## Required Features (28 total)
1. LIMIT_BAL
2. SEX
3. EDUCATION
4. MARRIAGE
5. AGE
6. PAY_0
7. PAY_2
8. PAY_3
9. PAY_4
10. PAY_5
11. PAY_6
12. BILL_AMT1
13. BILL_AMT2
14. BILL_AMT3
15. BILL_AMT4
16. BILL_AMT5
17. BILL_AMT6
18. PAY_AMT1
19. PAY_AMT2
20. PAY_AMT3
21. PAY_AMT4
22. PAY_AMT5
23. PAY_AMT6
24. AVG_PAYMENT_STATUS
25. MAX_PAYMENT_DELAY
26. TOTAL_BILL_AMT
27. TOTAL_PAY_AMT
28. UTILIZATION_RATIO

## Feature Engineering Required
Before prediction, you must create these features:
- AVG_PAYMENT_STATUS: Average of PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- MAX_PAYMENT_DELAY: Maximum of PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- TOTAL_BILL_AMT: Sum of BILL_AMT1 through BILL_AMT6
- TOTAL_PAY_AMT: Sum of PAY_AMT1 through PAY_AMT6
- UTILIZATION_RATIO: TOTAL_BILL_AMT / (LIMIT_BAL + 1)

## Usage Example
```python
import joblib
import pickle
import pandas as pd

# Load model and artifacts
model = joblib.load('best_credit_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Prepare input data with feature engineering
# ... (see streamlit_app.py for complete example)

# Scale and predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)
```

## Dependencies
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost (if using XGBoost model)

## Target Variable
- **Name:** default.payment.next.month
- **Classes:** 0 (No Default), 1 (Default)

## Important Notes
1. All input features must be scaled using the provided scaler
2. Feature engineering must be applied before scaling
3. Feature order must match exactly as in feature_names.pkl
4. Missing values are not expected - handle them before prediction

Generated: 2025-11-03 10:15:28
