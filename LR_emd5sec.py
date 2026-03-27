# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:50:36 2024

@author: Dania
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('emd5sec.csv')

# Extract features and target
X = data[['IMF_mean', 'IMF_std', 'IMF_energy', 'IMF_max', 'IMF_min', 'IMF_median', 'IMF_skewness', 'IMF_kurtosis']].values
Y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

reg = LogisticRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))