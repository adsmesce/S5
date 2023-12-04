import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

data = pd.read_csv("heart.csv") 
label_encoder = LabelEncoder()
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])

X = data.drop("HeartDisease", axis=1)

y = data["HeartDisease"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\nLinear where Constants = 1")
svm_model_linear = SVC(kernel='linear', C=1)
svm_model_linear.fit(X_train, y_train)

y_pred = svm_model_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


print("\n\nLinear where Constants = 100")
svm_model_linear = SVC(kernel='linear', C=100)
svm_model_linear.fit(X_train, y_train)

y_pred = svm_model_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


print("\n\RBF where Constants = 2")
svm_model_rbf = SVC(kernel='rbf', C=2)
svm_model_rbf.fit(X_train, y_train)

y_pred = svm_model_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


print("\n\RBF where Constants = 6")
svm_model_rbf = SVC(kernel='rbf', C=6)
svm_model_rbf.fit(X_train, y_train)

y_pred = svm_model_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


