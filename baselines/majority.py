import pandas as pd
from sklearn.metrics import accuracy_score

# loading the data
df = pd.read_csv("google-research/goemotions/data/train.tsv", sep="\t")

X = df['text']
y = df['emotion']  # adjust this depending on column names

# splitting into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# majority class
from collections import Counter
most_common = Counter(y_train).most_common(1)[0][0]
y_pred = [most_common] * len(y_test)

# evaluating
acc = accuracy_score(y_test, y_pred)
print(f"Majority Class Accuracy: {acc:.4f}")
