# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:06:15 2021

@author: Muhammad Salman Kabir
@purpose: Feature_Engineering (Filter Type) 
@regarding: Functional Connectivity Analysis
"""

# Importing necassary libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Importing necassary functions
from Classification import classification

# Top K features
K = 140

# Importing feature table
feature_table = pd.read_pickle("StackedFeatureTable\Feature_Table_Narrow_Beta_Band.pkl")

# Extracting features and class
features1 = feature_table.iloc[:,:-1]
# classes1 = feature_table.iloc[:,-1]

# Importing feature table
feature_table = pd.read_pickle("StackedFeatureTable\Feature_Table_Narrow_Theta_Band.pkl")

# Extracting features and class
features2 = feature_table.iloc[:,:-1]
classes = feature_table.iloc[:,-1]


features = pd.concat([features1,features2],axis=1)
# classes = pd.concat([classes1,classes2],axis=1)

feature_table = pd.concat([features,classes],axis=1)

# SelectKBest init and fitting
feature_engg = SelectKBest(score_func=chi2, k=100).fit(features,classes)

# Displaying top features with sc
scores = np.round(pd.DataFrame(feature_engg.scores_),4)
columns = pd.DataFrame(features.columns)
feature_scores = pd.concat([columns,scores],axis=1)
feature_scores.columns = ['Feature','Score']  

# Extracting top K features
top_features = feature_scores.nlargest(K,'Score')
    
# Forming feature table based on top features
best_feature_table =  pd.concat([feature_table[top_features.Feature],feature_table.Class],axis=1).to_numpy()
    
# Import shuffled indexes and shuffled feature table
shuffle = np.load("Shuffled_Indexes.npy")
best_feature_table = best_feature_table[shuffle,:]
    
# Computing accuracy
accuracy = classification(best_feature_table,20) 

# Mean and Standard Deviation   
print("\nMean Accuracy: %.2f \n" % np.mean(accuracy))
print("Standard Deviation: %.2f \n" % np.std(accuracy))