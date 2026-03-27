# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:53:35 2024

@author: Dania
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Read the data
data = pd.read_csv('emd5sec.csv')

# Extract features and target
X = data[['IMF_mean', 'IMF_std', 'IMF_energy', 'IMF_max', 'IMF_min', 'IMF_median', 'IMF_skewness', 'IMF_kurtosis']].values
Y = data['Label'].values

# Split the data into train and test sets with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
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
