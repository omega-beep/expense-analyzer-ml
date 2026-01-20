import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


data = {
    "text": [
        # Food
        "pizza hut order",
        "restaurant dinner",
        "burger king meal",
        "grocery store purchase",
        "zomato food delivery",

        # Shopping
        "amazon shopping",
        "amazon order",
        "flipkart purchase",
        "online shopping payment",
        "mall shopping",

        # Travel
        "uber ride",
        "bus ticket",
        "flight ticket booking",
        "train reservation",
        "ola cab ride",

        # Bills
        "electricity bill payment",
        "mobile recharge",
        "internet bill",
        "water bill",
        "credit card bill",

        # Entertainment
        "amazon prime",
        "netflix subscription",
        "spotify premium",
        "movie ticket",
        "concert booking",
        "gaming subscription"
    ],
    "category": [
        "Food","Food","Food","Food","Food",
        "Shopping","Shopping","Shopping","Shopping","Shopping",
        "Travel","Travel","Travel","Travel","Travel",
        "Bills","Bills","Bills","Bills","Bills",
        "Entertainment","Entertainment","Entertainment","Entertainment","Entertainment","Entertainment"
    ]
}


df = pd.DataFrame(data)

X = df["text"]
y = df["category"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    C=1
)

model.fit(X_vec, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved")
