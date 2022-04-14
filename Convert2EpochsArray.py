# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 00:08:51 2021

@author: Muhammad Salman Kabir
@purpose: To save given EEG data in Epoch_Arrays Object 
@regarding: Functional Connectivity Analysis 
"""

# Importing necassary libraries
import os
import mne
import shutil
import warnings
import numpy as np

# ignoring warning
warnings.filterwarnings("ignore")

# Making directory for storing epoch arrays
dir = "EpochArray"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

# Loading the given data
eeg_data = np.load('EEG_data.npy')


# Channels/Sensors names
ch_names = ["O2","O1","P4","P3","C4","C3","F4","F3","Fp2","Fp1",
            "P8","P7","T8","T7","F8","F7","Oz","Pz","Cz","Fz",
            "Fpz","FT7","FC3","FCz","FC4","FT8","TP7","CP3","CPz","CP4",
            "TP8"]

# Sampling frequency
sfreq = 250

# Info subject contianing sensor and samling frequency info
info = mne.create_info(ch_names,sfreq)


# Saving in Epoch Array format
for subject in range(eeg_data.shape[0]):
    print("\nSubject: "+str(subject+1))
    
    # Experiment have two conditions: Beginning and Ending
    cond_1 = eeg_data[subject,0,:,:,:]
    cond_2 = eeg_data[subject,1,:,:,:]
    
    # Concatination of sensor values w.r.t conditions
    data = np.vstack((cond_1,cond_2)) 
    
    # Creating Epoch Array Object
    epochs_array = mne.EpochsArray(data, info)
    
    # Plotting Epoch Array
    epochs_array.plot(picks=ch_names)
    
    # Saving in .fif format
    filename = "EpochArray\Subject_"+ str(subject+1) +"-epo.fif" 
    epochs_array.save(filename,overwrite=True)
    print("----------------------------------------")

