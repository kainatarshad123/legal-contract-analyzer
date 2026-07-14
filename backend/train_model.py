import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "ml_model", "clause_risk_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "risk_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml_model", "vectorizer.pkl")


# 1. Load dataset
data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print(data.head())


# 2. Separate input and output
X = data["clause_text"]
y = data["risk_level"]


# 3. Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 5. Train ML model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# 6. Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7. Save model and vectorizer into backend/ml_model
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nModel saved as:", MODEL_PATH)
print("Vectorizer saved as:", VECTORIZER_PATH)


# 8. Quick test
sample_clause = ["The tenant shall indemnify the landlord against all claims."]
sample_vector = vectorizer.transform(sample_clause)
sample_prediction = model.predict(sample_vector)

print("\nSample prediction:")
print(sample_clause[0])
print("Predicted risk:", sample_prediction[0])