# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:46:55 2024

@author: Dania
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('dwt5sec.csv')

# Extract features and target
X = data[['DWT_mean', 'DWT_std', 'DWT_energy', 'DWT_median', 'DWT_skewness', 'DWT_kurtosis']].values
Y = data['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled, y_train)

# Predictions on the training set
y_train_pred = svm_classifier.predict(X_train_scaled)

# Predictions on the test set
y_test_pred = svm_classifier.predict(X_test_scaled)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
