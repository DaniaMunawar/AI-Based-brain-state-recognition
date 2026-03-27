# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:54:08 2024

@author: Dania
"""

import pandas as pd
import numpy as np
from PyEMD import EMD

# Load the Excel file, skipping the second row
file_path = '25secwithlabel.xlsx'  # Replace 'your_excel_file.xlsx' with the path to your Excel file
data = pd.read_excel(file_path, skiprows=[1])

# Initialize EMD
emd = EMD()

# Placeholder for the output
output = np.zeros((len(data.columns), 3))  # Each row will hold mean, std, and energy

# Apply EMD on each column
for i, col in enumerate(data.columns):
    column_data = data[col].values
    IMFs = emd(column_data)
    
    # Extract features from the IMFs
    mean_imf = np.mean(IMFs)
    std_imf = np.std(IMFs)
    energy_imf = np.sum(IMFs**2)
    
    output[i, :] = [mean_imf, std_imf, energy_imf]

# Create a DataFrame from the output
columns = ['IMF_mean', 'IMF_std', 'IMF_energy']
emd_df = pd.DataFrame(output, columns=columns, index=data.columns)

# Save the EMD features to a new Excel file
output_path = 'emd25secfin.xlsx'  # Update with your desired output file path
emd_df.to_excel(output_path)

print(f"EMD features have been saved to {output_path}")
