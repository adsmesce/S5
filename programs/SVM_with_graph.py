import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
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

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


svm_model_linear = SVC(kernel='linear', C=1)
svm_model_linear.fit(X_train, y_train)

y_pred = svm_model_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


svm_model_rbf = SVC(kernel='rbf', C=1)
svm_model_rbf.fit(X_train, y_train)

y_pred = svm_model_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

svm_model_poly = SVC(kernel='poly', C=1)
svm_model_poly.fit(X_train, y_train)

y_pred = svm_model_poly.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_decision_regions(X_pca, y.to_numpy(), clf=svm_model_linear, legend=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Regions Linear')
plt.show()

plot_decision_regions(X_pca, y.to_numpy(), clf=svm_model_poly, legend=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Regions Poly')
plt.show()


plot_decision_regions(X_pca, y.to_numpy(), clf=svm_model_rbf, legend=2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Regions RBF')
plt.show()


