# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:16:54 2020

@author: 20202407
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode
from sklearn.neighbors import NearestCentroid
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def border_evaluation(mask):       
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 1)
    
    length_border = np.sum(border == 255)
    area = np.sum(mask > 0)
    
    return length_border, area

def imsize_evaluation(lesion, mask):
    x, y, z = lesion.shape
    size_score = x*y
    return size_score

def colour_evaluation(lesion, mask):
    mask_inv = 255 - mask
    colour_score = 0

    colours = {'light brown low':(255*0.588, 255*0.2, 255*0),
              'light brown high':(255*0.94, 255*0.588, 255*392),
              'dark brown low':(255*0.243, 255*0, 255*0),
              'dark brown high':(255*56, 255*0.392, 255*392),
              'white low':(255*0.8, 255*0.8, 255*0.8),
              'white high':(255, 255, 255),
              'red low':(255*0.588, 255*0, 255*0),
              'red high':(255, 255*0.19, 255*0.19),
              'blue gray low':(255*0, 255*0.392, 255*0.490),
              'blue gray high':(255*0.588, 255*0.588, 255*0.588),
              'black low':(255*0, 255*0, 255*0),
              'black high':(255*0.243, 255*0.243, 255*0.243)}

    for i in range(0,len(colours),2):
        mask_colour = cv2.inRange(lesion, colours.get(list(colours.keys())[i]), colours.get(list(colours.keys())[i+1]))
    
        if list(colours.keys())[i] == list(colours.keys())[-2] and list(colours.keys())[i+1] == list(colours.keys())[-1]:
            mask_colour = mask_colour - mask_inv
        
        if (np.sum(mask_colour > 0) / np.sum(mask > 0)) >= 0.05:    
            colour_score += 1
        
    
        
    return colour_score

def color_cluster_evaluation(lesion, mask, cluster_aantal = 5, HSV = False):
    
    if HSV == True:
        lesion = cv2.cvtColor(lesion, cv2.COLOR_RGB2HSV)
    
    w, h, d = tuple(lesion.shape) # Sla afmetingen op
    image_array = np.reshape(lesion, (w * h, d)) # Zet alle pixels onder elkaar
    image_array_sample = shuffle(image_array, random_state=0)[:10000] # Alleen de eerste random 10000
    # pixels worden meegenomen in het bepalen van de clusters zodat het niet een uur duurt
    
    kmeans = KMeans(n_clusters=cluster_aantal, random_state=0).fit(image_array_sample)
    # Maak clusters
    
    centroids = kmeans.cluster_centers_
    # Zwaartepunten of midden van clusters

    D = cdist(centroids, centroids, metric='seuclidean') # Afstand tussen alle midden clusters
    totaal = 0
    for i in range(cluster_aantal):
        for j in range(cluster_aantal):
            if i<j:
                totaal = totaal + D[i, j] # Gemiddelde afstand berekenen
                
    return totaal/((cluster_aantal*(cluster_aantal-1))/2)
    

def image_rotation(lesion, mask):
    # Niet meer in gebruik, vervangen door img_conversion
    
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

def img_conversion(mask, lesion):
    # De plaatjes worden al ingelezen en omgezet naar de goede formaten
    # lesion = cv2.bitwise_and(lesion, lesion, mask=mask) # Deze geeft een error dus toch mijn eigen oplossing
    
    masker_bool = mask[:, :]==0 # Maak het masker een boolean ding
    lesion2 = lesion.copy() # Error fixen, geen idee waarom
    lesion2[masker_bool] = [0, 0, 0] # Haal de rand weg
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    try:
        angle = cv2.fitEllipse(contours[0])[2] - 90
        
    except:
        angle = 0
        
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))
    
    return (mask, lesion)

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
    
    if superior.shape[0] == inferior_flipped.shape[0]:
            
        resultaat_horizontal = inferior_flipped - superior
    
    if left_flipped.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left_flipped.shape[1]-right.shape[1], 0, None, None)
            
        resultaat_vertical = left_flipped - right
        
    if left_flipped.shape[1] < right.shape[1]:
        left_flipped = cv2.copyMakeBorder(left_flipped, None, None, None, right.shape[1]-left_flipped.shape[1], 0, None, None)
            
        resultaat_vertical = right - left_flipped
        
    if left_flipped.shape[1] == right.shape[1]:
            
        resultaat_vertical = right - left_flipped
    
    pix_diff_vertical = np.sum(resultaat_vertical > 0)
    pix_diff_horizontal = np.sum(resultaat_horizontal > 0)
    
    return pix_diff_horizontal, pix_diff_vertical # Alle tussenstappen niet returnen, is alleen onnodig datagebruik

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
    D = cdist(X_test_val, X_train, metric='euclidean')
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

def nearest_mean_classifier(X_train, y_train, X_validation, X_test):
    # Returns the labels for test_data, predicted by the nearest mean classifier trained on X_train and y_train
    # Input:
    # X_train - num_train x num_features matrix with features for the training data
    # y_train - num_train x 1 vector with labels for the training data
    # X_validation - num_test x num_features matrix with features for the validation data
    # X_test - num_test x num_features matrix with features for the test data
    # Output:
    # y_pred_validation - num_test x 1 predicted vector with labels for the validation data
    # y_pred_test - num_test x 1 predicted vector with labels for the test data

    # Niet in gebruik

    X_test_val = np.vstack((X_validation, X_test))
    # Gooi datasets samen

    clf = NearestCentroid()
    clf.fit(X_train, y_train) # Bepaal de means
    
    predicted_labels = clf.predict(X_test_val) # Voorspel de data

    # Sla voorspellingen op
    y_pred_validation = predicted_labels[:len(X_validation)]
    y_pred_test = predicted_labels[len(X_validation):]
    return y_pred_validation, y_pred_test

def k_bepalen(X_train, y_train, X_test, y_test):
    error = []

    # Niet in gebruik, succes rate wordt bepaald

    for i in range(1, len(X_train)):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(X_train)), error, 'ro', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')