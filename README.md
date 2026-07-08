# Credit Card Fraud Detection System

A hybrid machine learning pipeline for real-time credit card fraud detection, deployed as a Flask web application.

## Overview
This system detects fraudulent and behaviorally anomalous transactions using a four-model pipeline combining supervised learning and unsupervised anomaly detection.

## Models Used
- Random Forest — AUC-ROC 0.9860, Fraud Recall 0.84
- XGBoost — AUC-ROC 0.9848, Fraud Recall 0.91
- Isolation Forest — unsupervised anomaly detection layer
- Meta Model (Stacking) — RF + XGBoost → Logistic Regression, AUC-ROC 0.9682, F1 0.74, Precision 0.71

## Pipeline Flow
Input Transaction → RF + XGBoost probabilities → Meta Model → Isolation Forest anomaly flag → Confidence Score (Safe / Suspicious / Fraud)

## Novel Contributions
- Three-layer detection combining supervised + unsupervised + stacking — rarely implemented together
- Behavioral feature engineering — customer-merchant distance and transaction timing for per-customer personalized detection
- Isolation Forest as second line of defense for novel fraud patterns missed by supervised models

## Dataset
Kaggle Simulated Credit Card Fraud Detection — 1.29M transactions, 0.579% fraud rate, handled via SMOTE

## Features
- Single transaction checker with real-time fraud confidence score
- Batch analyst dashboard — upload CSV, get color coded fraud monitoring table

## Tech Stack
Python, scikit-learn, XGBoost, Flask, pandas, numpy, imbalanced-learn

## Setup
```bash
pip install -r requirements.txt
python app.py
```
Note: Model files (pkl/json) are not included in this repo due to size limits. Retrain using the Colab notebook or contact for model files.
