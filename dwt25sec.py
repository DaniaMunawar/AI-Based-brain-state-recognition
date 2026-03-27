# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:26:00 2024

@author: Dania
"""

import pandas as pd
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

# Load the Excel file, skipping the second row
data = pd.read_csv('25secwithlabel.csv')

wavelet = 'db4'  
level = 5  

# Placeholder for the output
output = np.zeros((len(data.columns), 8))  # Each row will hold mean, std, energy, max, min, median, skewness, kurtosis

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
output_path = 'dwt25sec.xlsx'  
dwt_df.to_excel(output_path)

print(f"DWT features have been saved to {output_path}")
