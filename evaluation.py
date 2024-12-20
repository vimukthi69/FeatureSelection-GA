import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load encoded features from .npy file
X = np.load('encoded_text.npy')

# Load the sentiment labels
df = pd.read_csv('dataset/processed_sentiment_data.csv')
y = df['sentiment']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA to reduce dimensions
n_components = 203  # Same number as the GA-selected features
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Logistic Regression model on PCA-reduced features
pca_model = LogisticRegression(max_iter=500)
pca_model.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_pca = pca_model.predict(X_test_pca)
y_proba_pca = pca_model.predict_proba(X_test_pca)[:, 1]

# Calculate metrics
pca_f1 = f1_score(y_test, y_pred_pca)
pca_accuracy = accuracy_score(y_test, y_pred_pca)
pca_auc = roc_auc_score(y_test, y_proba_pca)

print("\nPCA with LR Metrics:")
print(f"F1 Score: {pca_f1:.4f}")
print(f"Accuracy: {pca_accuracy:.4f}")
print(f"AUC: {pca_auc:.4f}")

# Train SVM model on PCA-reduced features
pca_model_svm = SVC(probability=True, kernel="linear", random_state=42)
pca_model_svm.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_pca_svm = pca_model_svm.predict(X_test_pca)
y_proba_pca_svm = pca_model_svm.predict_proba(X_test_pca)[:, 1]

# Calculate metrics
pca_svm_f1 = f1_score(y_test, y_pred_pca_svm)
pca_svm_accuracy = accuracy_score(y_test, y_pred_pca_svm)
pca_svm_auc = roc_auc_score(y_test, y_proba_pca_svm)

print("\nPCA with SVM Metrics:")
print(f"F1 Score: {pca_svm_f1:.4f}")
print(f"Accuracy: {pca_svm_accuracy:.4f}")
print(f"AUC: {pca_svm_auc:.4f}")

# Train logistic regression model without feature selection
print("\nLR - Without Feature selection")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("F1 score : ", f1)
print("Accuracy : ", accuracy)
print("AUC : ", auc)

# Train SVM model without feature selection
print("\nSVM - Without Feature selection")
model_svm = SVC(probability=True, kernel="linear", random_state=42)
model_svm.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)
y_proba_svm = model_svm.predict_proba(X_test)[:, 1]

f1_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_proba_svm)

print("F1 score : ", f1_svm)
print("Accuracy : ", accuracy_svm)
print("AUC : ", auc_svm)
