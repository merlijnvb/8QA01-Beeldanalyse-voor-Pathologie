# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:50:14 2020

@author: 20203167
"""


import math
import pandas as pd
from tqdm import tqdm
import random
import csv
import Groep_05_functions as util
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
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

def extract_info(value_data):
    
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
        colour_score = (float(colour_score))  #-1.8)*3

        
        border_score = (border**2) / (area*math.pi*4)
        symmetry_score = (symmetry_vertical + symmetry_horizontal) / area
        
        list_border_score.append(border_score)
        list_symmetry_score.append(symmetry_score)
        list_colour_scores.append(colour_score)
        
    df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                        "Border score":list_border_score,
                        "Colour score":list_colour_scores}) 
    return df

def print_accuracy(test_features,control_group,folds,classifiers):
    from sklearn.model_selection import train_test_split

    from sklearn.neighbors import KNeighborsClassifier

    
    x_train, x_test, y_train, y_test = train_test_split(test_features, control_group, random_state=folds)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)   
  
    
    def get_accuracy(x,y):

        c = knn.score(x, y)

        return (c)

    training_sets = []
    test_sets = []
    
    for i in range(len(classifiers)):
        train = float(get_accuracy(x_train,y_train))
        test = float(get_accuracy(x_test, y_test))
        
        training_sets.append(train)
        test_sets.append(test)
       
    training_sets = tuple(training_sets)
    test_sets = tuple(test_sets)
    
    return (training_sets,test_sets)
    
def define_score():    
    value_data, control_group = read_files()
    productC1 = list(range(10,200, 189))
    print(productC1)
    productS1= list(range(70,260, 189))
    productB1 = list(range(30,220, 189))
    productC = []
    productB = []
    productS = []
    percentages= []
    for i in range(len(productC1)):
        productC1[i] = productC1[i] * 0.1
        productS1[i] = productS1[i] * 0.1
        productB1[i] = productB1[i] * 0.01

    for kleurfactor in productC1:
        for x in productS1:
            for z in productB1:
                productC.append(kleurfactor)
                productS.append(x)
                productB.append(z)
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
                    colour_score = ((float(colour_score))  -1.8) * float(kleurfactor)
            
                    
                    border_score = (border**2) / (area*math.pi*4) *float(z)
                    symmetry_score = (symmetry_vertical + symmetry_horizontal) / area * float(x)
                    
                    list_border_score.append(border_score)
                    list_symmetry_score.append(symmetry_score)
                    list_colour_scores.append(colour_score)
                    
                df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                                    "Border score":list_border_score,
                                    "Colour score":list_colour_scores})

                kf = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=1)
                # Predict labels for each fold using the KNN algortihm
                X = df.to_numpy() # Vanaf 2 zodat melanoma info niet wordt meegenomen
                # Load labels
                control_group = np.array(control_group)
                y = control_group

                for train_index, test_val_index in kf.split(X, y):
                    

                    
                    test_index , val_index = np.array_split(test_val_index, 2)
                    # np.array(X[train_index])
                    # np.array(y[train_index])
                    # np.array(y[val_index])
                    # np.array(y[test_index])
                    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
                    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
                    y_pred_val, y_pred_test_curr = util.knn_classifier(X_train, y_train, X_val, X_test, 5)
                    Curr_Acc = accuracy_score(y_val,y_pred_val)

                #Measure test set and store these values
                    test_acc_curr = accuracy_score(y_test, y_pred_test_curr)
                    # classifiers = ["Nearest Neighbor"]
                    

                    
                    # mean_train = []
                    # mean_test = []
                
                    # for train in data[:1]:
                    #     train = "{:0.2f}".format(train)
                    #     mean_train.append(train)
                        
                    # for test in data[1:]:
                    #     test = "{:0.2f}".format(test)
                    #     mean_test.append(test)

                    # print(test_acc_curr)
                    percentages.append(test_acc_curr)    
                    # print(data)
                

               

    return productC, productS, productB, percentages
x,y,z,percentages = define_score()

print(percentages)
print(x)
print(y)
print(z)
maximum = max(percentages)
minimum = min(percentages)
index = percentages.index(maximum)
print("Hoogste percentage:", maximum, 'met Kleur factor = ', x[index], ', Symmetrie factor = ', y[index], 'en Border Factor = ', z[index] )
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')





ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('ProductC')
ax.set_ylabel('ProductS')
ax.set_zlabel('ProductB')
c = percentages

img = ax.scatter(x, y, z, c=c, vmin= minimum, vmax = maximum,  cmap=plt.hot())


fig.colorbar(img)
plt.show()

# # mean_table = pd.DataFrame({"Mean training:":[mean_train],
# #                         "Mean test:":[mean_test]})
    
# # mean_table = mean_table.to_csv("classifiers_cluster.csv",index=False,sep=",")
# # with open('classifiers_cluster.csv', 'r') as file:
# #     reader = csv.reader(file)
# #     for row in reader:
# #         print(row)