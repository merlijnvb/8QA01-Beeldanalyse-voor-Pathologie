# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:03:20 2020

@author: 20203167
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import math
import pandas as pd
import Groep_05_functions as util
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
def read_files():
    csv = open("labels.csv")
    csv_read = csv.readlines()
    csv.close()
    results = open("results.csv")
    results_read = results.readlines()
    results.close()

    
    control_group = []
    value_data = []
    
    for lines in csv_read[1:]:
        lines = tuple(lines.rstrip().split(","))

        control_group.append(lines[1])

    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(";"))
        value_data.append(lines)
        
    return value_data, control_group



value_data, control_group = read_files()
   
list_colour_scores = []
list_border_score = []
list_symmetry_score = []

    
for tupl in value_data:
        
    border = int(tupl[1]) 
    area = int(tupl[2])
    symmetry_vertical = int(tupl[3])
    symmetry_horizontal = int(tupl[4])
    colour_score = tupl[-2]
    colour_score = colour_score.replace(",", ".")
    colour_score = ((float(colour_score))) 
    

    
    border_score = (border**2) / (area*math.pi*4) 
    symmetry_score = (symmetry_vertical + symmetry_horizontal) / area 
    
    list_border_score.append(border_score)
    list_symmetry_score.append(symmetry_score)
    list_colour_scores.append(colour_score)
    
df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                    "Border score":list_border_score,
                    "Colour score":list_colour_scores})


# Predict labels for each fold using the KNN algortihm
X = X_full = df.to_numpy() # Vanaf 2 zodat melanoma info niet wordt meegenomen
# Load labels
control_group = np.array(control_group)
y = control_group
distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X)),
]

for i in distributions:
    titel = i[0]
    X = i[1]

    symmetrie = X[:, 0]
    kleur = X[:, 2]
    border = X[:, 1]
    kf = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    for train_index, test_val_index in kf.split(X, y):
        

        
        test_index , val_index = np.array_split(test_val_index, 2)
        X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
        


        y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
        y_pred_val, y_pred_test_curr = util.knn_classifier(X_train, y_train, X_val, X_test, 4)
        print(y_pred_val)
        test_acc_curr = accuracy_score(y_test, y_pred_test_curr)
        print(titel, 'has accuracy', test_acc_curr)
    kleuren = []

    for i in y:
        if i == "True":
            kleuren.append('r')
        elif i == 'False':
            kleuren.append('b')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(titel)
    ax.scatter(kleur, border, symmetrie, c=kleuren, marker='o')
