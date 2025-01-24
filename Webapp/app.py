#!/usr/bin/env python3
from flask import Flask, render_template, request
import joblib
from scrapping import get_tweets, initialize_client, TextPreprocessor
from flask.cli import AppGroup
import pandas as pd
from twikit import Client
import praw
import asyncio
import numpy as np
from Reddit_posts import *
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze/", methods=["POST"])
def analyze():
    """ """
    topic = request.form.get("topic")
    print(f"Received topic: {topic}")  # Debug print

    if not topic:
        print("No topic provided")  # Debug print
        return render_template("index.html", error="Please enter a topic")
    try:
        get_reddit_posts(topic)
        print("Reddit posts fetched")  # Debug print
        data = pd.read_csv("posts.csv")
        print(f"Data loaded: {len(data)} rows")  # Debug print
        html_data = data.to_html(index=False)
        return render_template("index.html", data=html_data, topic=topic)
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        return render_template("index.html", error=str(e))



if __name__ == "__main__":
    app.run(debug=True)
