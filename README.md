# 😀 SMM: Social Media Mood 🙁
## Description
Web application that allows users to search a topic and check general opinion's sentiment around it (positive, negative and neutral)

## 🔗 Links:
- [Linkedin](https://www.linkedin.com/in/rahifhiba/)

## 📚 Directory structure:

```
Directory structure:
└── rahifhiba-smm-social-media-mood/
    ├── README.md
    ├── model.joblib
    ├── notebook.ipynb
    ├── notebook_env.db
    ├── requirements.txt
    ├── train.csv
    └── Webapp/
        ├── Reddit_posts.py
        ├── TextPreprocessing.py
        ├── app.py
        ├── posts.csv
        ├── requirements.txt
        ├── models/
        │   ├── sentiment_analysis_82%.joblib
        │   └── tfidf_vectorizer.pkl
        ├── static/
        │   ├── images/
        │   └── styles/
        │       └── style.css
        └── templates/
            ├── base.html
            ├── base_error.html
            ├── index.html
            └── table.html
```
## 💿 Installation:
for  nootbook:

```
pip install -r requirements.txt

```
for web application:

```
cd Webapp ;
pip install -r requirements.txt

```

## ⚙️Usage

```
git clone https://github.com/Rahifhiba/SMM-Social-Media-Mood/commits/main/
cd Webapp
./app.py
```
## ✍️ Authors

- [@Rahifhiba](https://www.github.com/Rahifhiba)


## License

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)

