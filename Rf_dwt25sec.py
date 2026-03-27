# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:39:36 2024

@author: Dania
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Read the data
data = pd.read_csv('dwt25sec.csv')

# Extract features and target
X = data[['DWT_mean', 'DWT_std', 'DWT_energy','DWT_median','DWT_skewness','DWT_kurtosis']].values
Y = data['Label'].values

# Split the data into train and test sets with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model with regularization parameters
rf_classifier = RandomForestClassifier(n_estimators=120, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predictions on the training set
y_train_pred = rf_classifier.predict(X_train_scaled)

# Predictions on the test set
y_test_pred = rf_classifier.predict(X_test_scaled)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy:", test_accuracy)
