import math
import pandas as pd
import matplotlib.pyplot as plt

def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    value_data = []
    
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines)
        
    return value_data

def make_grids():
    value_data = read_files()
    
    list_colour_scores = []
    list_border_score = []
    list_symmetry_score = []
    list_diameter_score = []
    list_colour_cluster_score = []

    list_border_length = []
    list_symmetry_hor_overlapse = []
    list_symmetry_ver_overlapse = []
    list_diameter = []
    list_area = []
    
    for tupl in value_data:
        border = int(tupl[1])
        area = int(tupl[2])
        symmetry_horizontal = int(tupl[3])
        symmetry_vertical = int(tupl[4])
        colour_score = int(tupl[5])
        colour_cluster_score = float(tupl[6])
        
        diameter = border / math.pi
        
        border_score = (border**2) / (area*math.pi*4)
        symmetry_score = (symmetry_vertical + symmetry_horizontal) / area
        diameter_score = diameter / area
        
        list_border_score.append(border_score)
        list_symmetry_score.append(symmetry_score)
        list_diameter_score.append(diameter_score)
        list_colour_scores.append(colour_score)
        list_colour_cluster_score.append(colour_cluster_score)

        list_area.append(area)
        list_border_length.append(border)
        list_diameter.append(diameter)
        list_symmetry_hor_overlapse.append(symmetry_horizontal)
        list_symmetry_ver_overlapse.append(symmetry_vertical)
        
    sdf = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                        "Border score":list_border_score,
                        "Colour score":list_colour_scores,
                        "Colour cluster":colour_cluster_score,
                        "Diameter score":list_diameter_score})  
    
    vdf = pd.DataFrame({"Area":list_area,
                        "Border":list_border_length,
                        "Diameter":list_diameter,
                        "Horizontal Symmetry":list_symmetry_hor_overlapse,
                        "Vertical Symmetry":list_symmetry_ver_overlapse,
                        "Colour score":list_colour_scores,
                        "Colour cluster":colour_cluster_score})  
    
    
    pd.plotting.scatter_matrix(sdf, hist_kwds={'bins':len(value_data)},diagonal='kde',figsize=(10,10))
    plt.suptitle("ABCD-scores plottetd",y=0.9125,fontsize=20)
    plt.savefig("ABCD_scatterd")
    
    pd.plotting.scatter_matrix(vdf, hist_kwds={'bins':len(value_data)},diagonal='kde',figsize=(10,10))
    plt.suptitle("Values plotted",y=0.9125,fontsize=20)
    plt.savefig("Values_scatterd")
    
make_grids()
