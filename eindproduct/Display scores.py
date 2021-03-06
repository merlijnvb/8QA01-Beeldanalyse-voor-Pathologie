# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:35:58 2021

@author: 20203167
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:03:20 2020

@author: 20203167
"""

#Dit bestand laat de Powertransformation(box-cox) zien. 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
import math
import pandas as pd
import Groep_05_functions as util
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
def read_files():
    csv = open("class2020_group05_labels.csv")
    csv_read = csv.readlines()
    csv.close()
    results = open("Resultaten_Groep_5.csv")
    results_read = results.readlines()
    results.close()

    
    control_group = []
    value_data = []
    
    for lines in csv_read[1:]:
        lines = tuple(lines.rstrip().split(","))

        control_group.append(lines[1])

    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines)
        
    return value_data, control_group



value_data, control_group = read_files()

list_colour_scores = []
list_border_score = []
list_symmetry_score = []

    
for tupl in value_data:
        
    border = float(tupl[2]) 
    area = float(tupl[1])
    symmetry_vertical = float(tupl[4])
    symmetry_horizontal = float(tupl[3])
    colour_score = tupl[5]
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
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X))
]




for i in distributions:

    titel = i[0]
    methode = i[0]

    X = i[1]
    X[:,0] = X[:,0]*1.6
    X[:,1] = X[:,1]*2.0
    X[:,2] = X[:,2]*2.0
    symmetrie = X[:, 0]
    kleur = X[:, 2]
    border = X[:, 1]
    kf = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    list_border_score.append(border_score)
    list_symmetry_score.append(symmetry_score)
    list_colour_scores.append(colour_score)
    for train_index, test_val_index in kf.split(X, y):
        

        
        test_index , val_index = np.array_split(test_val_index, 2)
        X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
        


        y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
        y_pred_val, y_pred_test_curr = util.knn_classifier(X_train, y_train, X_val, X_test, 4)
        test_acc_curr = accuracy_score(y_test, y_pred_test_curr)

    print(test_acc_curr)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
     
    kleuren = []
        
    for z in y:
        if z == "True":
            kleuren.append('r')
        elif z == 'False':
            kleuren.append('b')
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Asymmetry')
    ax.set_ylabel('Border')
    ax.set_zlabel('Color')
    plt.title(titel)
    ax.scatter(kleur, border, symmetrie, c=kleuren, marker='o')
