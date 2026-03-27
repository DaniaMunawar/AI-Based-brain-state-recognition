# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:47:49 2024

@author: Dania
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
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

# Initialize and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(140, 50), max_iter=600, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predictions on the training set
y_train_pred = mlp.predict(X_train_scaled)

# Predictions on the test set
y_test_pred = mlp.predict(X_test_scaled)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
