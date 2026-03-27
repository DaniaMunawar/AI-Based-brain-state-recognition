# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:12:03 2024

@author: Dania
"""
import pandas as pd
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

# Load the dataset
data = pd.read_csv('rawdataset.csv')

wavelet = 'db4'
level = 5

# Placeholder for the output
output = np.zeros((len(data.columns), 6))  # Each row will hold mean, std, energy, median, skewness, kurtosis

# Apply DWT on each column
for i, col in enumerate(data.columns):
    column_data = data[col].values
    coeffs = pywt.wavedec(column_data, wavelet, level=level)
    
    # Flatten the coefficients for feature extraction
    coeffs_flat = np.hstack(coeffs)
    
    # Extract features from the DWT coefficients
    mean_coeff = np.mean(coeffs_flat)
    std_coeff = np.std(coeffs_flat)
    energy_coeff = np.sum(coeffs_flat**2)
    median_coeff = np.median(coeffs_flat)
    skewness_coeff = skew(coeffs_flat)
    kurtosis_coeff = kurtosis(coeffs_flat)
    
    output[i, :] = [mean_coeff, std_coeff, energy_coeff, median_coeff, skewness_coeff, kurtosis_coeff]

# Create a DataFrame from the output
columns = ['DWT_mean', 'DWT_std', 'DWT_energy', 'DWT_median', 'DWT_skewness', 'DWT_kurtosis']
dwt_df = pd.DataFrame(output, columns=columns, index=data.columns)

# Save the DWT features to a new Excel file
output_path = 'dwt5sec.xlsx'
dwt_df.to_excel(output_path)

print(f"DWT features have been saved to {output_path}")
