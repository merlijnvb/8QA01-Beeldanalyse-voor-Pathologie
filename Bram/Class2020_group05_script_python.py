# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:00:10 2020

@author: 20202407
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Groep_05_functions as util
from skimage import morphology
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from tqdm import tqdm

dataFile = 'class2020_group05_labels.xlsx'
imagePath = 'C:\\Users\Administrator\Documents\OGO beeldanalyse\Onze data'
maskPath = 'C:\\Users\Administrator\Documents\OGO beeldanalyse\Onze data maskers'

dframe1 = pd.read_excel(dataFile)

#print(dframe)

ID = list(dframe1['id'])
Melanoma = np.array(dframe1['melanoma'])
numImages = len(ID)
featuresBorder = np.empty([numImages,1])
featuresBorder[:] = np.nan
featuresAsymmetry = np.empty([numImages,1])
featuresAsymmetry[:] = np.nan
featuresColorClusters = np.empty([numImages,1])
featuresColorClusters[:] = np.nan

for i in tqdm(np.arange(numImages)):
    
    # Define filenames related to this image
    imFile = imagePath + os.sep + ID[i] + '.jpg'
    maskFile = maskPath + os.sep + ID[i] + '_segmentation.png'
    
    # Read the images with these filenames
    im = plt.imread(imFile)
    mask = plt.imread(maskFile)
    #Draai de foto    
    mask = mask.astype(np.uint8)
    mask, im = util.img_conversion(mask, im)
    
    # Measure features
    border_score = util.border_evaluation(mask)
    color_score_clusters = util.color_cluster_evaluation(im, mask, cluster_aantal = 5, HSV = False) # Kleur met clusters
    asymmetry_score = util.symmetry_evaluation(im, mask)
    # xx, yy, zz = util.measureYourOwnFeatures(mask)
    
    # Store in the variables we created before
    featuresBorder[i] = border_score * 0.25
    featuresColorClusters[i] = color_score_clusters * 2
    featuresAsymmetry[i] = ((asymmetry_score[0]+asymmetry_score[1])/2) * 100
    # featuresOther[i,0] = xx
    
outfile = 'group2020_05_automatic.csv'
outdata = {"id": ID, 
           "melanoma": Melanoma.flatten(),
           "border": featuresBorder.flatten(),
           "asymmetry": featuresAsymmetry.flatten(),
           "colorclusters": featuresColorClusters.flatten()}

dframe_out = pd.DataFrame(outdata)
dframe_out.to_csv(outfile, index=False)

# Load the data you saved, then do some analysis
outfile = 'group2020_05_automatic.csv'
dframe = pd.read_csv(outfile)
ID = list(dframe['id']) # Leuk dat deze gegevens weer worden geladen maar buiten het plotten worden ze niet gebruikt
featuresBorder = np.array(dframe['border'])
featuresColor = np.array(dframe['asymmetry'])
featuresColorClusters = np.array(dframe['colorclusters'])

# Display 2 features measured in a scatterplot, 
axs = util.scatter_data(featuresAsymmetry, featuresColorClusters, Melanoma)
axs.set_xlabel('X1 = asymmetry')
axs.set_ylabel('X2 = Color Clusters')
axs.legend()

#Define K's that are tested on the validation set and the number of the current fold
K_range_lower = 2
K_range_upper = 20 # Not inclusive
Validation_K = range(K_range_lower, K_range_upper)
curr_fold = 0
# Load features
X = dframe.iloc[:,2:].to_numpy() # Vanaf 2 zodat melanoma info niet wordt meegenomen
# Load labels
y = Melanoma

K_range_length = K_range_upper - K_range_lower

all_acc_test= np.empty([5, K_range_length])
all_acc_test[:] = np.nan
all_acc_val=np.empty([5, K_range_length])
all_acc_val[:]=np.nan

# Split dataset into 5 different dataset folds for cross-validation
kf = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=1)
# Predict labels for each fold using the KNN algortihm
for train_index, test_val_index in kf.split(X, y):
    
    #For making a plot of test acc and val acc over multiple K
    acc_test_list = [] #Commented because average is plotted
    acc_val_list = []
    
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
        
        #Measure test set and store these values
        test_acc_curr = accuracy_score(y_test, y_pred_test_curr)
        acc_test_list.append(test_acc_curr)
        acc_val_list.append(Curr_Acc)
        
        # If accuracy of the predictions on the validation set is larger than the current accuracy, save predictions
        # for test set        
        if Curr_Acc > Acc_Score:
            Acc_Score = Curr_Acc
            y_pred_test = y_pred_test_curr
            best_K = K
    
    # Save accuracies of test and validation for calculating average
    all_acc_val[curr_fold, :] = acc_val_list
    all_acc_test[curr_fold, :] = acc_test_list
    
    # Add 1 to the number of the current fold and print the accuracy on the test set for the current fold
    curr_fold += 1
    test_acc = accuracy_score(y_test,y_pred_test)
    print('Accuracy of predictions on test set of fold '+ str(curr_fold)+ ': ' + str(test_acc))
    print('Accuracy of validation set was '+ str(Acc_Score) + ' with K: '+str(best_K))

#Plot average accuracies for all K's 
gem_acc_test=[] # Initialise variables for storing average accuracies
gem_acc_val=[]
for i in range(K_range_length):
    acc_val = all_acc_val[:, i] # Take the accuracies of a certain K
    acc_test = all_acc_test[:, i]
    gem_acc_test.append(np.sum(acc_test)/len(acc_test)) # Calculate and store average accuracy
    gem_acc_val.append(np.sum(acc_val)/len(acc_val))
#plt.plot(Validation_K, gem_acc_test,'r-') # Plot these average accuracies for each K
#plt.plot(Validation_K, gem_acc_val, 'b-') # Omdat ik niet weet hoe ik met plotten om moet gaan
# plot het allebei in 1