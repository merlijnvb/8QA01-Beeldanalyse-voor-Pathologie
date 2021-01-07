# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:00:10 2020

@author: 20202407
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Groep_05_functions as util
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

dataFile = 'Class2020_group05_labels.xlsx'
imagePath = 'C:\\Users\Administrator\Documents\OGO beeldanalyse\ISIC-2017_All_Data'
maskPath = 'C:\\Users\Administrator\Documents\OGO beeldanalyse\ISIC-2017_All_Data_GroundTruth'

dframe1 = pd.read_excel(dataFile)


ID = list(dframe1['id'])
Melanoma = np.array(dframe1['melanoma'])
numImages = len(ID)
featuresBorder = np.empty([numImages,1])
featuresBorder[:] = np.nan
featuresAsymmetryH = np.empty([numImages,1])
featuresAsymmetryH[:] = np.nan
featuresColorClusters = np.empty([numImages,1])
featuresColorClusters[:] = np.nan
featuresAsymmetryV = np.empty([numImages,1])
featuresAsymmetryV[:] = np.nan
featuresArea = np.empty([numImages,1])
featuresArea[:] = np.nan
featuresColor = np.empty([numImages,1])
featuresColor[:] = np.nan

for i in tqdm(np.arange(numImages)):
    
    # Define filenames related to this image
    imFile = imagePath + os.sep + ID[i] + '.jpg'
    maskFile = maskPath + os.sep + ID[i] + '_segmentation.png'
    
    # Read the images with these filenames
    im = plt.imread(imFile)
    mask = plt.imread(maskFile)
    #Rotate the image and mask
    mask = mask.astype(np.uint8)
    mask, im = util.img_conversion(mask, im)
    
    # Measure features
    border, area = util.border_evaluation(mask)
    color_score_clusters = util.color_cluster_evaluation(im, mask, cluster_aantal = 5, HSV = False)
    asymmetry_horizontal, asymmetry_vertical = util.symmetry_evaluation(im, mask)
    
    # Store in the variables we created before
    featuresBorder[i] = border
    featuresArea[i] = area
    featuresColorClusters[i] = color_score_clusters
    featuresAsymmetryH[i] = asymmetry_horizontal
    featuresAsymmetryV[i] = asymmetry_vertical
    
outfile = 'group2020_05_automatic.csv'
outdata = {"id": ID, 
           "border": featuresBorder.flatten(),
           "area": featuresArea.flatten(),
           "asymmetry horizontal": featuresAsymmetryH.flatten(),
           "asymmetry vertical": featuresAsymmetryV.flatten(),
           "colorclusters": featuresColorClusters.flatten()}

dframe_out = pd.DataFrame(outdata)
dframe_out.to_csv(outfile, index=False)
    
# Load the data you saved, then do some analysis
outfile = 'group2020_05_automatic.csv'
dframe = pd.read_csv(outfile)
ID = list(dframe['id'])
featuresBorder = np.array(dframe['border'])
featuresAsymmetryH = np.array(dframe['asymmetry horizontal'])
featuresAsymmetryV = np.array(dframe['asymmetry vertical'])
featuresColorClusters = np.array(dframe['colorclusters'])
featuresArea = np.array(dframe['area'])

# Display 2 features measured in a scatterplot, 
#axs = util.scatter_data(featuresAsymmetryH, featuresColorClusters, Melanoma)
#axs.set_xlabel('X1 = asymmetry')
#axs.set_ylabel('X2 = Color Clusters')
#axs.legend()
    
#Define K's that are tested on the validation set and the number of the current fold
K_range_lower = 2
K_range_upper = 20 # Not inclusive
Validation_K = range(K_range_lower, K_range_upper)
K_range_length = K_range_upper - K_range_lower

n_folds = 5
curr_fold = 0

# Load features
X_data = dframe.iloc[:,1:].to_numpy()

X_unscaled = np.empty([numImages, 3])
X_unscaled[:] = np.nan
for i in range(X_data.shape[0]):
    border_score = (X_data[i, 0]**2) / (X_data[i, 1]*math.pi*4) 
    symmetry_score = (X_data[i, 2] + X_data[i, 3])/X_data[i, 1]
    color_score = X_data[i, 4]   
    X_unscaled[i, 0] = border_score
    X_unscaled[i, 1] = symmetry_score
    X_unscaled[i, 2] = color_score

# Normalise features
X = PowerTransformer(method='box-cox').fit_transform(X_unscaled)
X[:, 0] = X[:, 0] * 2
X[:, 1] = X[:, 1] * 2
X[:, 2] = X[:, 2] * 1.6

# Load labels
y = Melanoma

# Split dataset into 5 different dataset folds for cross-validation
kf = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.4, random_state=1)
# Predict labels for each fold using the KNN algortihm
for train_index, test_val_index in kf.split(X, y):
        
    best_K = 0
    # Define accuracy score and predictions for test set
    Acc_Score = 0
    y_pred_test = 0
    # split dataset into a train, validation and test dataset
    test_index , val_index = np.split(test_val_index, 2)
    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
    # Generate predictions using knn_classifier for every K
    for K in Validation_K: 
        y_pred_val, y_pred_test_curr = util.knn_classifier(X_train, y_train, X_val, X_test, K)
        Curr_Acc = accuracy_score(y_val,y_pred_val)
        
        # If accuracy of the predictions on the validation set is larger than the current accuracy, save predictions
        # for test set        
        if Curr_Acc > Acc_Score:
            Acc_Score = Curr_Acc
            y_pred_test = y_pred_test_curr
            best_K = K
            
    
    # Add 1 to the number of the current fold and print the accuracy on the test set for the current fold
    curr_fold += 1
    test_acc = accuracy_score(y_test,y_pred_test)
    print('Accuracy of predictions on test set of fold '+ str(curr_fold)+ ': ' + str(test_acc))
    #print('Accuracy of validation set was '+ str(Acc_Score) + ' with K: '+str(best_K))