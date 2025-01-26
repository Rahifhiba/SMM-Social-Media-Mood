#!/usr/bin/env python3

from dotenv import load_dotenv
import os
import pandas as pd
import praw
from langdetect import detect, LangDetectException
from TextPreprocessing import TextPreprocessor
load_dotenv()

def is_eng(text):
    """checks if text is in english"""
    try:
        # Combine title and text for better detection accuracy
        combined = text if len(text) > 50 else text * 3
        return detect(combined) == 'en'
    except LangDetectException:
        return False

def clean_posts(posts):
    """clean posts"""
    posts = posts.apply(lambda x: x.replace("\n", " "))
    preprocessor = TextPreprocessor()
    posts = posts.apply(preprocessor.preprocess)
    return posts


def get_reddit_posts(search_query):
    """fetches posts from reddit related to query"""
    reddit = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT"),
    )

    subreddit = reddit.subreddit("all")
    posts = subreddit.search(search_query)

    posts_data = []
    for post in posts:
        if post.selftext:
            content = f"{post.title} {post.selftext}".strip()
            if is_eng(content):
                posts_data.append(
                    {
                        "author": post.author.name if post.author else "[Deleted]",
                        "title": post.title,
                        "text": post.selftext,
                    }
                )

    df_posts = pd.DataFrame(posts_data)
    if df_posts.empty:
        print("No posts found")
        return False
    df_posts["cleaned_text"] = clean_posts(df_posts["text"])
    df_posts.dropna( inplace=True)
    if df_posts.empty:
        print("All posts were filtered out during cleaning")
        return False
    print("NaNs after cleaning:", df_posts["cleaned_text"].isna().sum())
    df_posts.to_csv("posts.csv", index=False)
    return True


