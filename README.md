# Expense Analyzer using Machine Learning

A machine learning–based web application that automatically categorizes expense descriptions using NLP techniques and visualizes spending patterns.

## Features
- Text-based expense categorization using TF-IDF + Logistic Regression
- Top-2 category predictions with confidence scores
- Batch prediction via CSV upload
- Category-wise summary analytics
- Interactive bar and pie charts (Chart.js)
- Export predictions as downloadable CSV

## Tech Stack
- Python
- Flask
- scikit-learn
- pandas
- HTML, CSS, JavaScript
- Chart.js

## Machine Learning Approach
- Text vectorization using TF-IDF with unigrams and bigrams
- Regularized Logistic Regression for multi-class classification
- Probability-based predictions for confidence estimation

## Input Format (CSV)
```csv
description
uber ride
amazon prime
pizza hut order
