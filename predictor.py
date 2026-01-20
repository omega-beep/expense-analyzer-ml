import joblib
import numpy as np

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_top_categories(description, top_n=2):
    vec = vectorizer.transform([description])

    probabilities = model.predict_proba(vec)[0]
    classes = model.classes_

    # Pair each class with its probability
    class_probs = list(zip(classes, probabilities))

    # Sort by probability (descending)
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Take top N and convert to %
    top_predictions = [
        (category, round(prob * 100, 2))
        for category, prob in class_probs[:top_n]
    ]

    return top_predictions
