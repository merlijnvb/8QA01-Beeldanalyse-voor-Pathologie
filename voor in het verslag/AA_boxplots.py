import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import math
import pandas as pd

def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_group = []
    value_data = []
       
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines[:-1])
        control_group.append(lines[-1])
    
    return value_data, control_group

def orginise_values(value_data):    
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
    
    return df

def normalise_data(data_list): 
    X = PowerTransformer(method='box-cox').fit_transform(data_list)
    X[:,0] = X[:,0]*2.0
    X[:,1] = X[:,1]*2.0
    X[:,2] = X[:,2]*1.6
    
    symmetry = X[:, 0]
    border = X[:, 1]
    colour = X[:, 2]
    
    list_symmetry = []
    list_border = []
    list_colour = []
    
    for i in range(len(X)):
        list_symmetry.append(symmetry[i])
        list_border.append(border[i])
        list_colour.append(colour[i])

    return list_symmetry, list_border, list_colour

def get_plots():
    value_data, control_group = read_files()
    
    symmetry, border, colour = normalise_data(orginise_values(value_data))
    
    true_list_border_score = []
    true_list_symmetry_score = []
    true_list_colour_cluster = []
    
    false_list_border_score = []
    false_list_symmetry_score = []
    false_list_colour_cluster = []
    
    for j in range(len(control_group)):
        if control_group[j] == "True":
            true_list_border_score.append(border[j])
            true_list_symmetry_score.append(symmetry[j])
            true_list_colour_cluster.append(colour[j])
                
        if control_group[j] == "False":
            false_list_border_score.append(border[j])
            false_list_symmetry_score.append(symmetry[j])
            false_list_colour_cluster.append(colour[j])
    
    def print_plots(true,false, name):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.boxplot([true,false])
            ax.set_xticklabels( ['melanoma','non-melanoma'] )
            plt.title(name)
            plt.savefig("{} group 6".format(name))
    
    for lists in [(true_list_border_score,false_list_border_score,"Border Score"),(true_list_symmetry_score,false_list_symmetry_score,"Asymmetry Score"),(true_list_colour_cluster,false_list_colour_cluster,"Color Score")]:
        print_plots(lists[0],lists[1],lists[2])

get_plots()
