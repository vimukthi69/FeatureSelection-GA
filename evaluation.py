import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# Load encoded features from .npy file
X = np.load('encoded_text.npy')

# Load the sentiment labels
df = pd.read_csv('dataset/processed_sentiment_data.csv')
y = df['sentiment']  # Convert to binary

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA to reduce dimensions
n_components = 220  # Same number as the GA-selected features
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a Logistic Regression model on PCA-reduced features
pca_model = LogisticRegression(max_iter=500)
pca_model.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_pca = pca_model.predict(X_test_pca)
y_proba_pca = pca_model.predict_proba(X_test_pca)[:, 1]

# Calculate metrics
pca_f1 = f1_score(y_test, y_pred_pca)
pca_accuracy = accuracy_score(y_test, y_pred_pca)
pca_auc = roc_auc_score(y_test, y_proba_pca)

print("PCA Metrics:")
print(f"F1 Score: {pca_f1:.4f}")
print(f"Accuracy: {pca_accuracy:.4f}")
print(f"AUC: {pca_auc:.4f}")
