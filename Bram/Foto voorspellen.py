# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:15:13 2021

@author: 20202407
"""

import Groep_05_functions as util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

"""
Read all the files and ask for the input of the user. These input files should
be in the same folder as the program. The extension of the input file should
also be entered.
"""

imname = input("Name of lesion image: ")
maskname = input("Name of mask image: ")

im = plt.imread(imname)
mask = plt.imread(maskname)

infile = "group2020_05_all_results.csv"
dframe = pd.read_csv(infile)
X_raw = dframe.iloc[:, 1:6].to_numpy() # Geen ID en intervalscore meenemen

labelFile = "All_labels.xlsx"
labelFrame = pd.read_excel(labelFile)
y = np.array(labelFrame['melanoma'])

"""
Calculate the values for the given image.
"""

mask = mask.astype(np.uint8)
mask, im = util.img_conversion(mask, im) # Dat omgedraaid t.o.v. andere functies is irritant

border, area = util.border_evaluation(mask)
color_cluster_score = util.color_cluster_evaluation(im, mask)
asymmetry_horizontal, asymmetry_vertical = util.symmetry_evaluation(im, mask)

"""
Calculate the scores for the given image and the known data.
"""

border_score = (border**2) / (area*math.pi*4)
asymmetry_score = (asymmetry_vertical + asymmetry_horizontal) / area

scores_input = [border_score, asymmetry_score, color_cluster_score]

X_input = np.array(scores_input)
X_input = np.reshape(X_input, (1,3))
X_dump = np.empty([1, 3]) # Deze is alleen nodig omdat ik te lui ben om de code te herschrijven

X_given = np.empty([X_raw.shape[0], 3])
X_given[:] = np.nan

# Calculate and store scores for given data
for i in range(X_raw.shape[0]):
    border_score_g = (X_raw[i, 0]**2) / (X_raw[i, 1]*math.pi*4) 
    color_cluster_score_g = X_raw[i, 4] # _g to not interfere
    asymmetry_score_g = (X_raw[i, 2] + X_raw[i, 3]) / X_raw[i, 1]
    Scores = [border_score_g, asymmetry_score_g, color_cluster_score_g]
    X_given[i, :] = Scores


"""
Classify the given image and print result.
"""

y_pred_foto, y_pred_dump = util.knn_classifier(X_given, y, X_input, X_dump, 5)

if y_pred_foto == 0:
    print("Deze foto bevat waarschijnlijk geen melanoom.")
elif y_pred_foto == 1:
    print("Deze foto bevat waarschijnlijk een melanoom.")