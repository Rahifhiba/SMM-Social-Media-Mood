import pandas as pd
import re
import emoji
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        # Initialize the word lemmatizer and the emojis dictionary
        self.word_lemmatizer = WordNetLemmatizer()
        self.emojis = {
            ':)': 'smile',
             ':-)': 'smile',
             ';d': 'wink',
             ':-E': 'vampire',
             ':(': 'sad',

            ':-(': 'sad',
             ':-<': 'sad',
             ':P': 'raspberry',
             ':O': 'surprised',

            ':-@': 'shocked',
             ':@': 'shocked',
            ':-$': 'confused',
             ':\\': 'annoyed',

            ':#': 'mute',
             ':X': 'mute',
             ':^)': 'smile',
             ':-&': 'confused',
             '$_$': 'greedy',

            '@@': 'eyeroll',
             ':-!': 'confused',
             ':-D': 'smile',
             ':-0': 'yell',
             'O.o': 'confused',

            '<(-_-)>': 'robot',
             'd[-_-]b': 'dj',
             ":'-)": 'sad smile',
             ';)': 'wink',
             ';D': 'wink',

            ';-)': 'wink',
             'O:-)': 'angel',
            'O*-)': 'angel',
            '(:-D': 'gossip',
             '=^.^=': 'cat',
             ':D':'smile',
        }

    def clean_text(self, text):
        """Clean text from unwanted elements (urls, mentions, hashtags,
        non-alphabetical characters) and handle CamelCase hashtags."""

        def split_camel_case(hashtag):
            """Split CamelCase hashtags into separate words."""
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', hashtag)
        text = text.strip()
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', lambda m: split_camel_case(m.group(1)), text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text

    def filter_non_english_words(self, text):
        """Remove all words that contain non-English characters."""
        pattern = r'\b(?:[a-zA-Z]+|[a-zA-Z_]+)\b'
        filtered = re.findall(pattern, text)
        return ' '.join(filtered)

    def reduce_len_text(self, text):
        """Reduce text repetitive letters """
        repeat_regexp = re.compile(r'(.)\1{2,}')
        return repeat_regexp.sub(r'\1', text)

    def lemmatize_text(self, text):
        """ Lemmatize text """
        return ' '.join(self.word_lemmatizer.lemmatize(word) for word in text.split())

    def handle_emojies(self, text):
        """Replace emojis with text"""
        text = emoji.demojize(text, delimiters=(" ", " "))
        for e, meaning in self.emojis.items():
            text = text.replace(e, meaning)
        return text

    def preprocess(self, text):
        """
        Preprocess using all the previous functions
        """
        replace_emojie = self.handle_emojies(text)
        cleaned_text = self.clean_text(replace_emojie)
        no_repetition = self.reduce_len_text(cleaned_text)
        lemmatized_text = self.lemmatize_text(no_repetition)
        clean = self.filter_non_english_words(lemmatized_text)
        return clean

