import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib, os


# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target  # 0 = malignant, 1 = benign

# Select 5 features
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]
X = df[selected_features]
y = df['diagnosis']


# Handle missing values (if any)
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

loaded_model = joblib.load('model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Predict the first sample
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)
print("Predicted:", prediction[0])

import joblib

project_folder = os.path.dirname(os.path.abspath(__file__))  # root folder

joblib.dump(model, os.path.join(project_folder, "model.joblib"))
joblib.dump(scaler, os.path.join(project_folder, "scaler.joblib"))
