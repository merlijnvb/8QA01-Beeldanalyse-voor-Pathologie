import cv2
import os
import symmetry_function as sym
import image_conversion as conv
import colour_function as col
import border_function as bor
import diameter_function as dia
import Haar_verwijderen as hv 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import pandas as pd
def read_files():
    lesion_img_list = []
    lesion_mask_list = []
    lesion_list = []
    
    for lesion in os.listdir("data"):
        if lesion.endswith(".jpg"):
            lesion_img_list.append(lesion)
            
    for mask in os.listdir("data_masks"):
        if mask.endswith(".png"):
            lesion_mask_list.append(mask)
            
    for i in range(len(lesion_img_list)):
        img = lesion_img_list[i]
        img_mask = lesion_mask_list[i]
        lesion_list.append((img,img_mask))
    return lesion_list

def image_evaluator(lesion,mask):
    #lesion_mask_rotated = conv.image_rotation(lesion,mask)[0]
    #lesion_image_rotated = conv.image_rotation(lesion,mask)[1]
    
    lesion_mask_rotated = mask
    lesion_image_rotated = lesion = hv.haarverwijderen(lesion)
        
    sym_score = 2.6 * sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[0] + sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[1]
    bor_score = 0.8 * bor.border_evaluation(lesion_mask_rotated)[0]
    col_score = 0.5 *col.colour_evaluation(lesion_image_rotated, lesion_mask_rotated)
    dia_score = 0.5 *dia.diameter_evaluation(lesion_mask_rotated)
    
    return (sym_score,bor_score,col_score,dia_score)

def return_results():
    data = read_files()
    read_file = open("results6.csv", "a")
    read_file.write(('ID, asymmetrie_score, border_score, kleur_score, diameter_score'))
    
    results_images = []
    
    for nr in tqdm(range(len(data))):
        ID = nr
        lesion = cv2.imread("data\{}".format(data[nr][0]))
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread("data_masks\{}".format(data[nr][1]), cv2.IMREAD_GRAYSCALE)
        lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
                
        sym_score = image_evaluator(lesion,mask)[0]
        bor_score = image_evaluator(lesion,mask)[1]
        col_score = image_evaluator(lesion,mask)[2]
        dia_score = image_evaluator(lesion,mask)[3]
        
        results_images.append((ID,sym_score,bor_score,col_score,dia_score))
       

    for result in results_images:
        line = "\n{0:d},{1:f},{2:f},{3:f},{4:f}".format(result[0],result[1],result[2],result[3],result[4])
        read_file.write(line)
        

    read_file.close()
    
return_results()

    
def scatter_data(X1, X2, Y, ax=None):
    # scatter_data displays a scatterplot of dataset X1 vs X2, and gives each point
    # a different color based on its label in Y

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'Class ' + str(i)
        ax.scatter(X1[idx2], X2[idx2], color=c, label=lbl)

    return ax



def knn_classifier(X_train, y_train, X_validation, X_test, k):
    # Returns the labels for test_data, predicted by the k-NN clasifier trained on X_train and y_train
    # Input:
    # X_train - num_train x num_features matrix with features for the training data
    # y_train - num_train x 1 vector with labels for the training data
    # X_validation - num_test x num_features matrix with features for the validation data
    # X_test - num_test x num_features matrix with features for the test data
    # k - Number of neighbors to take into account
    # Output:
    # y_pred_validation - num_test x 1 predicted vector with labels for the validation data
    # y_pred_test - num_test x 1 predicted vector with labels for the test data

    X_test_val = np.vstack((X_validation, X_test))
    # Compute standardized euclidian distance of validation and test points to the other points
    D = cdist(X_test_val, X_train, metric='seuclidean')
    # Sort distances per row and return array of indices from low to high
    sort_ix = np.argsort(D, axis=1)
    # Get the k smallest distances
    sort_ix_k = sort_ix[:, :k]
    predicted_labels = y_train[sort_ix_k]
    # Predictions for each point is the mode of the K labels closest to the point
    predicted_labels = mode(predicted_labels, axis=1)[0]
    y_pred_validation = predicted_labels[:len(X_validation)]
    y_pred_test = predicted_labels[len(X_validation):]
    return y_pred_validation, y_pred_test        

dframe = pd.read_csv('C:/Users/20203167/Documents/GitHub/8qa01/templates/class2020_group01_labels.csv')

ID = list(dframe['image_id'])
Melanoma = np.array(dframe['melanoma'])

 
outfile = 'C:/Users/20203167/Documents/results6.csv'
dframe = pd.read_csv(outfile)

ID = list(dframe['ID'])
arrayID = dframe[['ID']].to_numpy()

featuresAsymmetry = dframe[[' asymmetrie_score']].to_numpy()

featuresBorder = dframe[[' border_score']].to_numpy()
featuresKleur = dframe[[' kleur_score']].to_numpy()
featuresDiameter = dframe[[' diameter_score']].to_numpy()
features = np.hstack((featuresAsymmetry, featuresDiameter, featuresKleur, featuresBorder))
# Display the features measured in a scatterplot

Validation_K = range(1, 20)
curr_fold = 0
# Load features
X = dframe.iloc[:,1:].to_numpy()
# Load labels
y = Melanoma
all_acc_test= np.empty([5, 19])
all_acc_test[:] = np.nan
all_acc_val=np.empty([5, 19])
all_acc_val[:]=np.nan

# Split dataset into 5 different dataset folds for cross-validation
kf = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=1)
# Predict labels for each fold using the KNN algortihm
for train_index, test_val_index in kf.split(X, y):
    
    #For making a plot of test acc and val acc over multiple K
    acc_test_list = [] 
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
        y_pred_val, y_pred_test_curr = knn_classifier(X_train, y_train, X_val, X_test, K)
        Curr_Acc = accuracy_score(y_val,y_pred_val)
        # If accuracy of the predictions on the validation set is larger than the current accuracy, save predictions
        # for test set
        
        #Measure test set and store these values
        test_acc_curr = accuracy_score(y_test,y_pred_test_curr)
        acc_test_list.append(test_acc_curr)
        acc_val_list.append(Curr_Acc)
        
        if Curr_Acc > Acc_Score:
            Acc_Score = Curr_Acc
            y_pred_test = y_pred_test_curr
            best_K = K
    # Add 1 to the number of the current fold and print the accuracy on the test set for the current fold
    #plt.plot(acc_test_list, 'r-') Plot accuracy for each fold
    #plt.plot(acc_val_list, 'b-') Commented because average is plotted
    all_acc_val[curr_fold, :] = acc_val_list
    all_acc_test[curr_fold, :] = acc_test_list
    curr_fold += 1
    plt.xlabel = "K"
    plt.ylabel = "Accuracy"
    test_acc = accuracy_score(y_test,y_pred_test)
    print('Accuracy of predictions on test set of fold '+ str(curr_fold)+ ': ' + str(test_acc))
    print('Accuracy of validation set was '+ str(Acc_Score) + ' with K: '+str(best_K))

#Plot average accuracies for all K's
gem_acc_test=[] # Initialise variables for storing average accuracies
gem_acc_val=[]
for i in range(19):
    acc_val = all_acc_val[:, i] # Take the accuracies of a certain K
    acc_test = all_acc_test[:, i]
    gem_acc_test.append(np.sum(acc_test)/len(acc_test)) # Calculate and store average accuracy
    gem_acc_val.append(np.sum(acc_val)/len(acc_val))
plt.plot(Validation_K, gem_acc_test,'r-') # Plot these average accuracies for each K
plt.plot(Validation_K, gem_acc_val, 'b-')