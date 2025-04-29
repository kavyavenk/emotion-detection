from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import baseline1_majority_class

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(baseline1_majority_class.X_train)
X_test_vec = vectorizer.transform(baseline1_majority_class.X_test)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, baseline1_majority_class.y_train)

# Predict
y_pred = clf.predict(X_test_vec)

# Evaluate
print(classification_report(baseline1_majority_class.y_test, y_pred))
