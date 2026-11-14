import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Data load karo
if os.path.exists("spam.csv"):
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    print("Asli dataset use ho raha hai:", len(df), "messages")
else:
    spam = [
        "Congratulations! You have won a free prize. Click here to claim",
        "URGENT! Your account is suspended. Verify now",
        "Win cash now!!! Reply YES to get 1000 dollars",
        "Free entry in a weekly contest. Text WIN to 80085",
        "You are selected for a lottery reward. Claim today",
        "Get cheap loans instantly with no documents",
        "Limited offer! Buy now and get 90% discount",
        "Your bank password expired. Click the link to update",
    ]
    ham = [
        "Hi, are we meeting for lunch tomorrow?",
        "Please find the project report attached",
        "Can you call me when you reach home?",
        "Meeting is shifted to 4 PM in the conference room",
        "Thanks for your help yesterday, really appreciate it",
        "Don't forget to submit the assignment by Friday",
        "Happy birthday! Have a great day",
        "I will send the notes in the evening",
    ]
    df = pd.DataFrame({
        "label": ["spam"] * len(spam) + ["ham"] * len(ham),
        "text": spam + ham,
    })
    print("Chhota sample data use ho raha hai (asli spam.csv nahi mila)")

df["target"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
)

# 3. Model banao aur train karo
model = make_pipeline(
    TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
    MultinomialNB(alpha=0.05),
)
model.fit(X_train, y_train)
# 4. Accuracy check karo
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred, target_names=["Normal", "Spam"]))

# 5. Model save karo
joblib.dump(model, "spam_model.joblib")
print("Model save ho gaya: spam_model.joblib")