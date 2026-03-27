# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:57:15 2024

@author: Dania
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Initialize and train the MLP model with regularization
mlp = MLPClassifier(hidden_layer_sizes=(140, 50), max_iter=500, random_state=42, alpha=0.02, early_stopping=True, validation_fraction=0.1)

# Fit the model
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
