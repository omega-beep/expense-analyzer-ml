# Smart Expense Analyzer (ML)

A machine learning–based web application that categorizes expense descriptions using NLP and performs amount-based expense analysis to show where money is spent.

## What This Project Does
- Classifies expense descriptions into categories using machine learning
- Analyzes both **single expenses** and **bulk transactions**
- Aggregates **total money spent per category**
- Visualizes spending patterns using charts
- Exports analyzed results as a CSV file

## Features
- NLP-based expense categorization (TF-IDF + Logistic Regression)
- Single expense analysis with amount and confidence score
- Batch expense analysis via CSV upload
- Category-wise transaction count and total spend
- Bar and pie chart visualizations (Chart.js)
- Downloadable CSV with predictions and amounts

## Tech Stack
- Python
- Flask
- scikit-learn
- pandas
- HTML, CSS, JavaScript
- Chart.js

## Machine Learning Approach
- Text vectorization using TF-IDF (unigrams + bigrams)
- Regularized Logistic Regression for multi-class classification
- Probability-based predictions for confidence estimation

## CSV Input Format
```csv
description,amount
uber ride,250
amazon prime,1499
pizza hut order,620
electricity bill,2100
