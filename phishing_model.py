import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import seaborn as sns
import matplotlib.pyplot as plt

def url_count(text):
    return len(re.findall(r'http[s]?://|www\\.', str(text)))

def special_char_count(text):
    return len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', str(text)))

df = pd.read_csv("dataset.csv")
df["text"] = df["text"].fillna("")

df["url_count"] = df["text"].apply(url_count)
df["special_char_count"] = df["text"].apply(special_char_count)

X_text = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_text_vec = vectorizer.fit_transform(X_text)

X = hstack([X_text_vec, df[["url_count", "special_char_count"]].values])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()
