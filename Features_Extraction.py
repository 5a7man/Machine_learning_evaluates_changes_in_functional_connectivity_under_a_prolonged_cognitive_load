# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:11:30 2021

@author: Muhammad Salman Kabir
@purpose: Feature Extraction 
@regarding: Functional Connectivity Analysis
"""
"""

# Importing necassary libraries
import os
import mne
import shutil
import warnings
import numpy as np
import scipy.signal as ss

# ignoring warning
warnings.filterwarnings("ignore")

def Connectivity_Matrix(epoch_object):
    conectivity_matrix = []
    
    # Extracting data from epoch object in array form
    data = epoch_object.get_data(picks='all')
    
    # No of epochs, channels and datapoints/epoch
    epochs = data.shape[0]
    channels = data.shape[1]
    
    # Predefining connectivity matrix for freq bands
    beta = np.zeros([epochs,channels,channels],dtype=float)
    narrow_beta = np.zeros([epochs,channels,channels],dtype=float)
    theta = np.zeros([epochs,channels,channels],dtype=float)
    narrow_theta = np.zeros([epochs,channels,channels],dtype=float)
    
    for epoch in range(epochs):
        # Extracting datapoints correspond to respective epoch
        data_points = data[epoch,:,:]
    
        for i in range(channels):
            for j in range(channels):
                # Computing coherence
                f, Cxy = ss.coherence(data_points[i,:],data_points[j,:],fs= 250)
                
                # Coherence for specific freq band
                beta[epoch,i,j] = np.mean(Cxy[np.where((f>=15) & (f<=30))])
                narrow_beta[epoch,i,j] = np.mean(Cxy[np.where((f>=22.5) & (f<=25))])
                theta[epoch,i,j] = np.mean(Cxy[np.where((f>=4) & (f<=8))])
                narrow_theta[epoch,i,j] = np.mean(Cxy[np.where((f>=4) & (f<=6.5))])
                    
            
    # Appending variables in one variable    
    conectivity_matrix.append(['Beta_Band',beta])
    conectivity_matrix.append(['Narrow_Beta_Band',narrow_beta])
    conectivity_matrix.append(['Theta_Band',theta])
    conectivity_matrix.append(['Theta_Narrow',narrow_theta])
    
    # Return connectivity matrix            
    return conectivity_matrix


def Feature_Table(conectivity_matrix):
    # Predefining feature table
    feature_table = []
    
    # frequency bands
    freq_bands = ["Beta_Band","Narrow_Beta_Band","Theta_Band","Narrow_Theta_Band"]
    
    
    for band in range(len(freq_bands)):
        # extracting connectivity matric for specific freq band
        matrix = conectivity_matrix[band][1]
        
        # Temporary table to hold features for single epoch
        table = np.zeros([matrix.shape[0],466])
        
        # extracting predictors/features
        for epoch in range(matrix.shape[0]):
            k = 0
            for i in range(matrix.shape[1]-1):
                for j in range(matrix.shape[2]-i-1):
                    table[epoch,k] = matrix[epoch,i,j+i+1] 
                    k = k+1
        
        # Class assignment -> 0: Early, 1: Late
        table[40:,465] = 1
        
        # forming feature table
        feature_table.append([freq_bands[band],table])
        
    return feature_table
        

####### Feature Extraction #######

# Making directories
dir = "ConnectivityMatrix"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

dir = "FeatureTable"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)


subjects = 20
for subject in range(subjects):
    # Loading subject's data
    print("\nSubject: " +str(subject+1))
    filename = "EpochArray\Subject_"+ str(subject+1) +"-epo.fif" 
    epoch_object = mne.read_epochs(filename, preload=False)
    
    # Computing and saving connectivity matrix 
    connectivity_matrix = Connectivity_Matrix(epoch_object)
    np.save("ConnectivityMatrix\Subject_"+str(subject+1)+".npy",connectivity_matrix)
    
    # Computing and  saving feature Table
    feature_table = Feature_Table(connectivity_matrix)
    np.save("FeatureTable\Subject_"+str(subject+1)+".npy",feature_table)
    
    print("---------------------------------------------")
    

    
    
    