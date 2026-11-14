# SpamGuard: Spam Email & SMS Detector

A Flask web app that classifies emails and SMS messages as spam or normal using machine learning
(TF-IDF + Multinomial Naive Bayes). Test accuracy: 99.0%, spam precision: 0.99, spam recall: 0.94.
Users can register, log in, check messages, see the words that indicate spam, and review their history.

## Features
- User registration, login and logout (hashed passwords, CSRF protection)
- Spam / normal prediction with confidence score
- Highlighted spam-indicating words
- Per-user history stored in SQLite

## Tech stack
Python, Flask, scikit-learn, pandas, SQLite, HTML/CSS

## Run locally
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. python train_model.py   (creates spam_model.joblib)
5. python server.py        (open http://127.0.0.1:5000)

## Note
Trained on the SMS Spam Collection dataset (English). Hindi/Hinglish accuracy is lower, and the model
misses some spam messages. Predictions are not always correct.