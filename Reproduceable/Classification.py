# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:18:44 2021

@author: Muhammad Salman Kabir
@purpose: k-folds classification using SVM classifier 
@regarding: Functional Connectivity Analysis
"""

## Importing libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import *

def classification(feature_table,k):
    ## ------------------------------------------------------------------------
    # classification do the SVM classification
    # Input -->
    #       feature_table: feature table in standard form
    #       k: folds for cross validation
    #
    # Output -->
    #       Accuracy
    ##-------------------------------------------------------------------------
    
    # Extracting predictors 
    features = (feature_table[:,:-1])
    
    # Extracting response
    classes = feature_table[:,-1]
        
    # SVM Classifier init.
    clf = sk.svm.SVC(kernel='rbf', random_state=42)
    
    # k fold init
    cv = sk.model_selection.KFold(n_splits=k)
    
    # Accuracy estimation
    accuracy = sk.model_selection.cross_val_score(clf, features, classes, scoring='accuracy', cv=cv)
    
    # Return accuracy
    return np.round(accuracy*100,2)


####### Main #######

#  Loading feature table
feature_table = pd.read_pickle("StackedFeatureTable\Feature_Table_Theta_Band.pkl").to_numpy()

# Import shuffled indexes and shuffled feature table
shuffle = np.load("Shuffled_Indexes.npy")
feature_table = feature_table[shuffle,:]

# Computing accuracy
accuracy = classification(feature_table,20)

# Mean and Standard Deviation 
print("\nMean Accuracy: %.2f \n" % np.mean(accuracy))
print("Standard Deviation: %.2f \n" % np.std(accuracy))
