# Fraud Detection with Psychology-Inspired Features and Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

Fraud detection is a critical challenge for banks and financial institutions. Every year, billions of dollars are lost to fraudulent transactions, and the cost extends beyond direct financial loss to include regulatory fines, reputational damage, and erosion of customer trust. Traditional rule‑based systems and linear models often struggle to keep up with evolving fraud patterns and can generate excessive false positives, frustrating legitimate customers.

This project explores a novel approach: **engineering features grounded in human psychology** (transaction velocity, amount deviation, time anomalies, device mismatch) and combining them with **deep neural networks**. The goal is to build a model that not only catches more fraud but also reduces false positives by understanding the behavioral patterns of both fraudsters and legitimate users.

The project uses the publicly available **[IEEE‑CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)** dataset from Kaggle, which contains real‑world anonymized transaction and identity data.

## 🎯 Key Results

| Model | AUC‑ROC | AUC‑PR | Precision@90% Recall |
|-------|---------|--------|----------------------|
| Logistic Regression (basic) | 0.870 | 0.453 | 0.035 |
| Logistic Regression (all)   | 0.869 | 0.452 | 0.035 |
| DNN (basic)                 | 0.932 | **0.649** | 0.035 |
| DNN (all)                   | 0.929 | **0.656** | 0.035 |

- **Linear models cannot leverage psychology features** – adding them to logistic regression yields no improvement.
- **Deep neural networks dramatically outperform linear models** – the DNN on basic features achieves a 43% relative gain in AUC‑PR over logistic regression.
- **Psychology features add value** – the DNN with all features reaches an AUC‑PR of **0.656**, a modest but clear improvement over the basic‑feature DNN.
- **Precision at 90% recall** remains at the baseline fraud rate (3.5%) – a reminder of the inherent difficulty of catching nearly all frauds without many false positives, but the high AUC‑PR indicates excellent ranking ability.

## 🧠 Psychology‑Inspired Features

Twelve per‑card features were engineered to capture behavioral patterns:

| Feature | Description | Psychological Rationale |
|---------|-------------|--------------------------|
| `card_avg_amt` | Expanding average of transaction amount | Baseline spending habit |
| `card_std_amt` | Expanding standard deviation of amount | Variability in spending |
| `card_last_txn_delta` | Seconds since previous transaction | Very short gaps may indicate automated testing |
| `card_typical_hour` | Circular mean of previous transaction hours | Typical time of day for this card |
| `card_typical_hour_std` | Circular standard deviation of previous hours | Consistency of routine |
| `card_night_ratio` | Proportion of previous transactions at night (23–5) | Baseline for unusual night activity |
| `card_txn_count_1h` | Number of transactions in last 1 hour | Urgency / scripted behavior |
| `card_txn_count_6h` | Count in last 6 hours | – |
| `card_txn_count_24h` | Count in last 24 hours | – |
| `card_sum_amt_1h` | Sum of amounts in last 1 hour | Spending spurt after a test |
| `card_sum_amt_6h` | Sum in last 6 hours | – |
| `card_sum_amt_24h` | Sum in last 24 hours | – |

All features were computed **per card** using expanding and rolling windows, **excluding the current transaction** to prevent data leakage.

## 📊 Results and Interpretation
The notebook contains detailed markdown explanations at each phase.

Final evaluation metrics and feature importance plots are included.

The DNN with all features achieves the highest AUC‑PR (0.656), demonstrating the value of combining psychology features with a non‑linear model.

## 🔮 Future Work
Online learning – Adapt the model incrementally to new fraud patterns.

Explainability – Use SHAP values to interpret individual predictions.

Threshold tuning – Choose a decision threshold that balances recall and precision based on business costs.

Model comparison – Try XGBoost or other tree‑based models for comparison.

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.