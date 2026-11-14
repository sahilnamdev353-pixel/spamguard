import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["target"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.25, random_state=42, stratify=df["target"])

rows = []
for alpha, stop, ngram, sub in itertools.product([1.0, 0.3, 0.1, 0.05], ["english", None], [(1, 1), (1, 2)], [False, True]):
    model = make_pipeline(
        TfidfVectorizer(stop_words=stop, ngram_range=ngram, sublinear_tf=sub),
        MultinomialNB(alpha=alpha))
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    for th in [0.5, 0.4, 0.3, 0.2]:
        pred = (proba >= th).astype(int)
        rows.append(dict(alpha=alpha, stop_words=stop or "none", ngram=f"{ngram[0]}-{ngram[1]}",
                         sublinear=sub, threshold=th,
                         accuracy=round(accuracy_score(y_test, pred) * 100, 2),
                         spam_precision=round(precision_score(y_test, pred, zero_division=0), 3),
                         spam_recall=round(recall_score(y_test, pred), 3),
                         spam_f1=round(f1_score(y_test, pred), 3)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

base = res[(res.alpha == 1.0) & (res.stop_words == "english") & (res.ngram == "1-1") & (~res.sublinear) & (res.threshold == 0.5)]
print("\nCURRENT MODEL (baseline):")
print(base.to_string(index=False))

good = res[res.spam_precision >= 0.95].sort_values(["spam_f1", "spam_recall"], ascending=False).head(8)
print("\nTOP 8 SETTINGS (spam precision stays >= 0.95):")
print(good.to_string(index=False))