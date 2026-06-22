# Phishing Email Detection Model

## Overview
This project is a machine learning model that detects whether an email is **Phishing or Safe** using text analysis and URL-based features.

## Features
- TF-IDF based email text processing
- URL detection feature extraction
- Special character analysis
- Logistic Regression classification model
- Accuracy evaluation
- Confusion matrix visualization

## Dataset Format
CSV file with two columns:
- text → email content
- label → 0 (Safe), 1 (Phishing)

## How to Run

```bash
pip install -r requirements.txt
python phishing_model.py
