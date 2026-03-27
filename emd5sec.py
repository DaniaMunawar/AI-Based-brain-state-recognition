5# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:11:34 2024

@author: Dania
"""

import pandas as pd
import numpy as np
from PyEMD import EMD
from scipy.stats import skew, kurtosis

# List of columns to process
columns_to_process = [
    'b8_13', 'b13_18', 'b18_23', 'b23_28', 'b28_33', 'b33_38', 'b38_43', 'b43_48',
    'b48_53', 'b53_58', 'b58_63', 'b65_70', 'r4_9', 'r9_14', 'r14_19', 'r19_24',
    'r24_29', 'r29_34', 'r34_39', 'r39_44', 'r8_13b', 'r13_18b', 'r18_23b', 'r23_28b',
    'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6', 'SB7', 'SB8', 'SR1', 'SR2', 'SR3',
    'SR4', 'SR5', 'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7', 'DB8', 'DB9',
    'DB10', 'DB11', 'DB12', 'DB13', 'DB14', 'AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6',
    'AB7', 'AB8', 'AR1', 'AR2', 'AR3', 'AR4', 'AR5', 'AR6', 'AR7', 'AR8', 'AR9', 'AR10',
    'AR11', 'AR12', 'AR13', 'IB1', 'IB2', 'IB3', 'IB4', 'IB5', 'IB6', 'IB7', 'IB8',
    'IB9', 'IB10', 'IB11', 'IB12', 'IB13', 'IB14', 'IB15', 'IB16', 'IB17', 'IB18',
    'IB19', 'IB20', 'IR1', 'IR2', 'IR3', 'IR4', 'IR5', 'IR6', 'IR7', 'IR8', 'IR9',
    'IR10', 'IR11', 'IR12', 'AsB1', 'AsB2', 'AsB3', 'AsB4', 'AsB5', 'AsB6', 'AsB7',
    'AsB8', 'AsB9', 'AsB10', 'AsB11', 'AsB12', 'AsB13', 'AsB14', 'AsR1', 'AsR2',
    'AsR3', 'AsR4', 'Ub1', 'Ub2', 'Ub3', 'Ub4', 'Ub5', 'Ub6', 'Ub7', 'Ub8', 'Ub9',
    'Ub10', 'Ub11', 'Ub12', 'UR1', 'UR2', 'UR3', 'UR4', 'UR5', 'UR6', 'UR7', 'UR8',
    'UR9', 'UR10', 'UR11', 'er1'
]

# Load the CSV file
data = pd.read_csv('rawdataset.csv', usecols=columns_to_process)

# Initialize EMD
emd = EMD()

# Placeholder for the output
output = np.zeros((len(columns_to_process), 8))  # Each row will hold mean, std, energy, max, min, median, skewness, kurtosis

# Function to clean the data
def clean_data(column_data):
    finite_data = column_data[np.isfinite(column_data)]
    if len(finite_data) == 0:
        return np.zeros(len(column_data))  # If no finite values, return an array of zeros
    return finite_data

# Apply EMD on each column
for i, col in enumerate(columns_to_process):
    column_data = data[col].values
    column_data = clean_data(column_data)  # Clean the data
    if len(column_data) == 0:
        continue  # Skip if no valid data
    
    IMFs = emd(column_data)
    
    # Flatten the IMFs for feature extraction
    imfs_flat = IMFs.flatten()
    
    # Extract features from the IMFs
    mean_imf = np.mean(imfs_flat)
    std_imf = np.std(imfs_flat)
    energy_imf = np.sum(imfs_flat**2)
    max_imf = np.max(imfs_flat)
    min_imf = np.min(imfs_flat)
    median_imf = np.median(imfs_flat)
    skewness_imf = skew(imfs_flat)
    kurtosis_imf = kurtosis(imfs_flat)
    
    output[i, :] = [mean_imf, std_imf, energy_imf, max_imf, min_imf, median_imf, skewness_imf, kurtosis_imf]

# Create a DataFrame from the output
columns = ['IMF_mean', 'IMF_std', 'IMF_energy', 'IMF_max', 'IMF_min', 'IMF_median', 'IMF_skewness', 'IMF_kurtosis']
emd_df = pd.DataFrame(output, columns=columns, index=columns_to_process)

# Save the EMD features to a new Excel file
output_path = 'emd5sec.xlsx'  
emd_df.to_excel(output_path)

print(f"EMD features have been saved to {output_path}")
