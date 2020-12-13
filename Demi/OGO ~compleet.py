import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode

def border_evaluation(mask):   
    height, width = mask.shape[:2]
    dim = (width-2, height-2)  

    resized = cv2.resize(mask, dim)
    resized = cv2.copyMakeBorder(resized, 1, 1, 1, 1, 0, None, None)
    border = mask - resized
    
    length_border = np.sum(border > 0)
    area_mask = np.sum(mask > 0)

    border_score = (length_border**2) / (4*math.pi*area_mask)
    
    return (border_score, border)

def colour_evaluation(lesion, mask):
    color_score = 0
    
    # intervals
    light_brown_higher_range = (249, 193, 160)
    light_brown_lower_range = (55, 24, 22)
    
    dark_brown_higher_range = (55, 24, 22)
    dark_brown_lower_range = (36, 15, 15)
    
    white_higher_range = (255, 255, 255)
    white_lower_range = (217, 217, 217)
    
    red_higher_range = (255, 77, 77)
    red_lower_range = (154, 0, 0)
    
    blue_grey_higher_range = (144, 168, 180)
    blue_grey_lower_range = (69, 91, 102)
    
    black_higher_range = (38, 38, 38)
    black_lower_range = (0, 0, 0)
    
    # mask of colour
    mask_light_brown = cv2.inRange(lesion, light_brown_lower_range, light_brown_higher_range)
    mask_dark_brown = cv2.inRange(lesion, dark_brown_lower_range, dark_brown_higher_range)
    mask_white = cv2.inRange(lesion, white_lower_range, white_higher_range)
    mask_red = cv2.inRange(lesion, red_lower_range, red_higher_range)
    mask_blue_grey = cv2.inRange(lesion, blue_grey_lower_range, blue_grey_higher_range)
    mask_black = cv2.inRange(lesion, black_lower_range, black_higher_range)
    
    # area colours
    pix_melanoma = np.sum(mask == 255)
    pix_light_brown = np.sum(mask_light_brown == 255)
    pix_dark_brown = np.sum(mask_dark_brown == 255)
    pix_white = np.sum(mask_white == 255)
    pix_red = np.sum(mask_red == 255)
    pix_blue_gray = np.sum(mask_blue_grey == 255)
    pix_black = np.sum(mask_black == 255) - ((lesion.shape[0] * lesion.shape[1]) - pix_melanoma)
    
    #counting system
    for i in [pix_dark_brown, pix_light_brown, pix_white, pix_red, pix_blue_gray, pix_black]:
        if (i/pix_melanoma) >= 0.05:
            color_score += 1
    print(color_score)
            
    return (color_score,mask_light_brown,mask_dark_brown,mask_white,mask_red,mask_blue_grey,mask_black)

def image_rotation(lesion, mask):
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    has_ellipse = len(contours) > 0
    
    if has_ellipse:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        angle = ellipse[2] - 90
        x, y = ellipse[1]
    
    
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, moment, (width, height))
    rotated_mask_scaled = cv2.copyMakeBorder(rotated_mask, 25, 25, 25, 25, 0, None, None)
    
    rotated_image = cv2.warpAffine(lesion, moment, (width, height))
    rotated_image_scaled = cv2.copyMakeBorder(rotated_image, 25, 25, 25, 25, 0, None, None)
    
    return (rotated_mask_scaled, rotated_image_scaled)

def symmetry_evaluation(lesion, mask):
    
    height, width = lesion.shape[:2]
      
    # calculate moments of binary image
    moment = cv2.moments(mask)
    
    # calculate x,y coordinate of center
    centre_x = int(moment["m10"] / moment["m00"])
    centre_y = int(moment["m01"] / moment["m00"])
    
    #lesion[ y1:y2   ,    x1:x2    ]
    superior = mask[0:centre_y, 0:width]
    inferior = mask[centre_y:height, 0:width]
    inferior_flipped = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_x]
    left_flipped = cv2.flip(left, 1)
    right = mask[0:height, centre_x:width]
    
        
    if superior.shape[0] > inferior_flipped.shape[0]:
        inferior_flipped = cv2.copyMakeBorder(inferior_flipped, superior.shape[0]-inferior_flipped.shape[0], None, None, None, 0, None, None)
                     
        resultaat_horizontal = superior - inferior_flipped
        
    if superior.shape[0] < inferior_flipped.shape[0]:
        superior = cv2.copyMakeBorder(superior, inferior_flipped.shape[0]-superior.shape[0], None, None, None, 0, None, None)
            
        resultaat_horizontal = superior - inferior_flipped
    
    if superior.shape[0] < inferior_flipped.shape[0]:
            
        resultaat_horizontal = inferior_flipped - superior
    
    if left_flipped.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left_flipped.shape[1]-right.shape[1], 0, None, None)
            
        resultaat_vertical = left_flipped - right
        
    if left_flipped.shape[1] < right.shape[1]:
        left_flipped = cv2.copyMakeBorder(left_flipped, None, None, None, right.shape[1]-left_flipped.shape[1], 0, None, None)
            
        resultaat_vertical = right - left_flipped
        
    if left_flipped.shape[1] == right.shape[1]:
            
        resultaat_vertical = right - left_flipped
    
    pix_melanoma = np.sum(lesion > 0)
    pix_diff_vertical = np.sum(resultaat_vertical > 0)
    pix_diff_horizontal = np.sum(resultaat_horizontal > 0)
    
    quotient_vertical = pix_diff_vertical / pix_melanoma
    quotient_horizontal = pix_diff_horizontal / pix_melanoma
    
    print(quotient_vertical)
    print(quotient_horizontal)
    
    return (quotient_vertical,quotient_horizontal,right,left_flipped,superior,inferior_flipped,resultaat_horizontal,resultaat_vertical)


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
    
    return 

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

dataFile = 'class2020_group05_labels.xlsx' 
imagePath = 'C:\\Users\public\OGO beeldanalyse\Onze data'
maskPath = 'C:\\Users\public\OGO beeldanalyse\Onze data maskers'

dframe = pd.read_excel(dataFile)

ID = list(dframe['id'])
Melanoma = np.array(dframe['Melanoma'])
numImages = len(ID)
featuresAsymmetry = np.empty([numImages,1])
featuresAsymmetry[:] = np.nan
featuresBorder = np.empty([numImages,1])
featuresBorder[:] = np.nan
featuresColor = np.empty([numImages,1])
featuresColor = np.nan


for i in np.arange(numImages):
    
    # Define filenames related to this image
    imFile = imagePath + os.sep + ID[i] + '.jpg'
    maskFile = maskPath + os.sep + ID[i] + '_segmentation.png'
    
    # Read the images with these filenames
    image = plt.imread(imFile)
    mask = plt.imread(maskFile)
    
    # Measure features
    border_score, border = border_evaluation(mask)
    color_score = colour_evaluation(image, mask)
    asymmetry_score = symmetry_evaluation(image, mask)
    
    # Store in the variables we created before
    featuresBorder[i,0] = border_score
    featuresColor[i,0] = color_score
    featuresAsymmetry[i,0] = asymmetry_score
    
outfile = 'group2019_XY_automatic.csv'
outdata = {"id": ID, 
           "border": featuresBorder.flatten(),
           "color": featuresColor.flatten(),
           "asymmetry": featuresAsymmetry()}

dframe_out = pd.DataFrame(outdata)
dframe_out.to_csv(outfile, index=False)

# Load the data you saved, then do some analysis
dframe = pd.read_csv(outfile)
ID = list(dframe["id"])
featuresBorder = np.array(dframe['border'])
featuresColor = np.array(dframe['color'])
featuresAsymmetry = np.array(dframe['asymmetry'])

# Display the features measured in a scatterplot
axs = scatter_data(featuresBorder, featuresColor, featuresAsymmetry, Melanoma)
axs.set_xlabel('X1 = Border')
axs.set_ylabel('X2 = Color')
axs.set_zlabel('X3 = Asymmetry')
axs.legend()

#Define K's that are tested on the validation set and the number of the current fold
Validation_K = range(1, 100)
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
