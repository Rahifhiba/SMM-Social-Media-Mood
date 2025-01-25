#!/usr/bin/env python3
""" main file web-app """

from flask import Flask, render_template, request
import joblib

# from scrapping import get_tweets, initialize_client, TextPreprocessor
from flask.cli import AppGroup
import pandas as pd
from twikit import Client
import praw
import numpy as np
from Reddit_posts import *

app = Flask(__name__)

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/sentiment_analysis_82%.joblib")


def classify_text(probabilities, threshold=0.3):
    """Classify text using the model"""
    labels = []
    for p in probabilities:
        neg = p[0]
        pos = p[1]
        if abs(neg - pos) <= threshold:
            labels.append(-1)
        else:
            labels.append(0 if pos < neg else 1)

    return labels


def classify(text):
    """classify text as positive, negative or neutral"""
    text = vectorizer.transform([text])
    probabilities = model.predict_proba(text)
    sentiment = classify_text(probabilities)
    return sentiment[0]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict():
    """classify text as positive, negative or neutral"""
    topic = request.form.get("topic")
    print(f"Received topic: {topic}")  # Debug print

    if not topic:
        print("No topic provided")  # Debug print
        return render_template("index.html", error="Please enter a topic")
    try:
        get_reddit_posts(topic)
        print("Reddit posts fetched")  # Debug print
        data = pd.read_csv("posts.csv")
        data.dropna(inplace=True)
        # Debugging prints
        print(f"Data loaded: {len(data)} rows")  # Debug print
        print("data's columns: ", data.columns)  # Debug print
        print("Na in data: \n", data.isna().sum())  # Debug print
        # End of debugging prints

        data["sentiment"] = data["cleaned_text"].apply(classify)
        data["sentiment"] = data["sentiment"].map(
            {0: "Negative", 1: "Positive", -1: "Neutral"}
        )

        positive_per = round((data["sentiment"] == "Positive").mean() * 100, 2)
        negative_per = round((data["sentiment"] == "Negative").mean() * 100, 2)
        neutral_per = round((data["sentiment"] == "Neutral").mean() * 100, 2)

        html_data = data.to_html(index=False)
        return render_template(
            "index.html",
            data=html_data,
            topic=topic,
            positive_percentage=positive_per,
            negative_percentage=negative_per,
            neutral_percentage=neutral_per,
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
