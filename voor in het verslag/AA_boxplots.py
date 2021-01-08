import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import math
import pandas as pd

"""
    READ THE RESULTS FILE AND SAVE VALUES IN LISTS
"""
def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_group = []
    value_data = []
    
    #PICK THE DATA FROM THE RESULTS FILE.
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines[:-1])
        control_group.append(lines[-1])
    
    return value_data, control_group

"""
    ORGANISE THE VALUES FURTHER AND MAKE SCORES.
"""
def orginise_values(value_data):    
    list_colour_scores = []
    list_border_score = []
    list_symmetry_score = []
    
    #FORMAT THE DATA AND MAKE SCORES.
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

"""
    NORMALISE THE SCORES WITH THE BOX-COX TRANSFORM FUNCTION. ==> ALL VALUES WILL BE IN THE RANGE [-6,6] EXCEPT THE COLOUR RANGE, THIS WILL BE [-3,3].
"""
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
    
    #CONVERT NUMPY LIST TO LISTS PER FEATURE
    for i in range(len(X)):
        list_symmetry.append(symmetry[i])
        list_border.append(border[i])
        list_colour.append(colour[i])

    return list_symmetry, list_border, list_colour

"""
    MAKE BOXPLOTS FROM THE NORMALISED SCORES
"""
def get_plots():
    #GATHER SCORES
    value_data, control_group = read_files() 
    
    #NORMALISE THE SCORES
    symmetry, border, colour = normalise_data(orginise_values(value_data))
    
    true_list_border_score = []
    true_list_symmetry_score = []
    true_list_colour_cluster = []
    
    false_list_border_score = []
    false_list_symmetry_score = []
    false_list_colour_cluster = []
    
    #SPLITS THE SCORES IN A NON-MELANOMA AND AN MELANOMA LIST
    for j in range(len(control_group)):
        if control_group[j] == "True":
            true_list_border_score.append(border[j])
            true_list_symmetry_score.append(symmetry[j])
            true_list_colour_cluster.append(colour[j])
                
        if control_group[j] == "False":
            false_list_border_score.append(border[j])
            false_list_symmetry_score.append(symmetry[j])
            false_list_colour_cluster.append(colour[j])
    
    """PLOTS THE LISTS"""
    def print_plots(true,false, name):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.boxplot([true,false])
            ax.set_xticklabels( ['melanoma','non-melanoma'] )
            plt.title(name)
            plt.savefig(name)
    
    #GET PLOTS
    for lists in [(true_list_border_score,false_list_border_score,"Border Score"),(true_list_symmetry_score,false_list_symmetry_score,"Asymmetry Score"),(true_list_colour_cluster,false_list_colour_cluster,"Color Score")]:
        print_plots(lists[0],lists[1],lists[2])

get_plots()
