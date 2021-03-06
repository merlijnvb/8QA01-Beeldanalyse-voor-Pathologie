# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:03:20 2020

@author: 20203167
"""

# Dit bestand test de verschillende factoren op verschillende normalizeringen en transformaties en print de beste opties en laat de percentages zien in een plot. 
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
#define a list of different normalizations/transformations. 
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



#loop over that list
for i in distributions:
    #create lists with all the color factors you want to try
    productC1 = list(range(1, 40, 1))
    productS1= list(range(1,40, 1))
    productB1 = list(range(1, 40, 1))
    productC = []
    productB = []
    productS = []
    percentages= []
    for a in range(len(productC1)):
        productC1[a] = productC1[a] * 0.1
        productS1[a] = productS1[a] * 0.1
        productB1[a] = productB1[a] * 0.1

    for kleurfactor in productC1:
        for symmetriefactor in productS1:
            for borderfactor in productB1:
                #loop over the list to try every possibility. Save the numbers to check which factor combination worked the best.
                titel = i[0]
                methode = i[0]
                productC.append(kleurfactor)
                productS.append(symmetriefactor)
                productB.append(borderfactor)
            
                X[:,0] = X[:,0]*symmetriefactor
                X[:,1] = X[:,1]*kleurfactor
                X[:,2] = X[:,2]*borderfactor
                symmetrie = X[:, 0]
                kleur = X[:, 2]
                border = X[:, 1]
                kf = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
                for train_index, test_val_index in kf.split(X, y):
                    
            
                    
                    test_index , val_index = np.array_split(test_val_index, 2)
                    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
                    
            
            
                    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
                    y_pred_val, y_pred_test_curr = util.knn_classifier(X_train, y_train, X_val, X_test, 4)
                    test_acc_curr = accuracy_score(y_test, y_pred_test_curr)
                    percentages.append(test_acc_curr)
                    #test het algoritme voor elke factor en sla de percentages voor de bijbehorende factor op. 
    maximum = max(percentages)
    minimum = min(percentages)
    index = percentages.index(maximum)
    print(i[0], "heeft als hoogste percentage:", maximum, 'met Kleur factor = ', productC[index], ', Symmetrie factor = ', productS[index], 'en Border Factor = ', productB[index] )
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Kleurfactor')
    ax.set_ylabel('Borderfactor')
    ax.set_zlabel('Symmetriefactor')
    img = ax.scatter(productC, productB,productS, c=percentages, vmin= minimum, vmax = maximum,  cmap=plt.hot())
    fig.colorbar(img)
    plt.show()        
    #plot het figuur waarbij de kleur de percentages aangeeft en de andere de bijbehorende factoren. Hierbij is vooral de geprinte tekst bruikbaar.