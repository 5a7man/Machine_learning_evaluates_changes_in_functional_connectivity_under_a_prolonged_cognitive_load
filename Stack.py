# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:17:58 2021

@author: Muhammad Salman Kabir
@purpose: Stacking features 
@regarding: Functional Connectivity Analysis
"""

# Importing necassary libraries
import os
import shutil
import numpy as np
import pandas as pd

# Making directory
dir = "StackedFeatureTable"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

# frequency bands
freq_bands = ["Beta_Band","Narrow_Beta_Band","Theta_Band","Narrow_Theta_Band"]

# Importing features names    
beta_band = np.load("Features_Names_Beta.npy")
theta_band = np.load("Features_Names_Theta.npy")

# Stacking
for band in range(len(freq_bands)):
    #Predefining stacked variable
    stacked_feature_table = np.zeros([1,466])
    print(freq_bands[band])
    
    subjects = 20
    for subject in range(subjects):
        # Loading feature table
        print("Subject: " +str(subject+1))
        filename = "FeatureTable\Subject_"+ str(subject+1) +".npy" 
        epoch_object = np.load(filename, allow_pickle=True)
        temp = np.load(filename,allow_pickle=True)[band,1]
        
        # Appending features for incoming subject
        stacked_feature_table = np.vstack([stacked_feature_table,temp])
        
        
    stacked_feature_table = stacked_feature_table[1:,:]
    
    if band <=1:
        df = pd.DataFrame(data=stacked_feature_table, columns=np.hstack([beta_band,"Class"]))
    else:
        df = pd.DataFrame(data=stacked_feature_table, columns=np.hstack([theta_band,"Class"]))
        
    df.to_pickle("StackedFeatureTable\Feature_Table_"+str(freq_bands[band])+".pkl")
    print("-------------------------------------------")