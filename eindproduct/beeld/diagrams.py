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

def make_plots():
    value_data = read_files()
    
    true_list_border_length = []
    true_list_symmetry_hor_overlapse = []
    true_list_symmetry_ver_overlapse = []
    true_list_symmetry_overlapse = []
    true_list_diameter = []
    true_list_area = []
    true_list_colour_score = []
    true_list_colour_cluster = []
    
    false_list_border_length = []
    false_list_symmetry_hor_overlapse = []
    false_list_symmetry_ver_overlapse = []
    false_list_symmetry_overlapse = []
    false_list_diameter = []
    false_list_area = []
    false_list_colour_score = []
    false_list_colour_cluster = []

    for tupl in value_data:
        border = int(tupl[1])
        area = int(tupl[2])
        diameter = border / math.pi
        symmetry_horizontal = int(tupl[3])
        symmetry_vertical = int(tupl[4])
        colour_score = int(tupl[5])
        colour_cluster_score = float(tupl[6])
        
        if tupl[7] == "True":
            true_list_border_length.append(border)
            true_list_symmetry_hor_overlapse.append(symmetry_horizontal)
            true_list_symmetry_ver_overlapse.append(symmetry_vertical)
            true_list_diameter.append(diameter)
            true_list_area.append(area)
            true_list_symmetry_overlapse.append(symmetry_horizontal+symmetry_vertical)
            true_list_colour_score.append(colour_score)
            true_list_colour_cluster.append(colour_cluster_score)
            
        if tupl[7] == "False":
            false_list_border_length.append(border)
            false_list_symmetry_hor_overlapse.append(symmetry_horizontal)
            false_list_symmetry_ver_overlapse.append(symmetry_vertical)
            false_list_diameter.append(diameter)
            false_list_area.append(area)
            false_list_symmetry_overlapse.append(symmetry_horizontal+symmetry_vertical)
            false_list_colour_score.append(colour_score)
            false_list_colour_cluster.append(colour_cluster_score)
    
    plt.plot(false_list_area,false_list_border_length,'b.', label="non-melanoma")
    plt.plot(true_list_area,true_list_border_length,'r.', label="melanoma")
    plt.xlabel("Area")
    plt.ylabel("Border Length")
    plt.title("Border-Area diagram")
    plt.legend()
    plt.savefig("Border-Area diagram")
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.boxplot([true_list_border_length,false_list_border_length],notch=True)
    # ax.set_xticklabels( ['melanoma','non-melanoma'] )
    
    plt.plot(false_list_area,false_list_diameter,'b.', label="non-melanoma")
    plt.plot(true_list_area,true_list_diameter,'r.', label="melanoma")
    plt.xlabel("Area")
    plt.ylabel("Diameter")
    plt.title("Diameter-Area diagram")
    plt.legend()
    plt.savefig("Diameter-Area diagram")
    plt.show()
    
    # plt.plot(false_list_area,false_list_symmetry_hor_overlapse,'b.', label="non-melanoma")
    # plt.plot(true_list_area,true_list_symmetry_hor_overlapse,'r.', label="melanoma")
    # plt.xlabel("Area")
    # plt.ylabel("Horizontal overlapse")
    # plt.title("Horizontal overlapse-Area diagram")
    # plt.legend()
    # plt.show()
    
    # plt.plot(false_list_area,false_list_symmetry_ver_overlapse,'b.', label="non-melanoma")
    # plt.plot(true_list_area,true_list_symmetry_ver_overlapse,'r.', label="melanoma")
    # plt.xlabel("Area")
    # plt.ylabel("Vertical overlapse")
    # plt.title("Vertical overlapse-Area diagram")
    # plt.legend()
    # plt.show()
    
    plt.plot(false_list_area,false_list_symmetry_overlapse,'b.', label="non-melanoma")
    plt.plot(true_list_area,true_list_symmetry_overlapse,'r.', label="melanoma")
    plt.xlabel("Area")
    plt.ylabel("Total overlapse")
    plt.title("Overlapse-Area diagram")
    plt.legend()
    plt.savefig("Overlapse-Area diagram")
    plt.show()
    
    plt.bar("0", true_list_colour_score.count(0), label="melanoma", color="b")
    plt.bar("1", true_list_colour_score.count(1), color="b")
    plt.bar("2", true_list_colour_score.count(2), color="b")
    plt.bar("3", true_list_colour_score.count(3), color="b")
    plt.bar("4", true_list_colour_score.count(4), color="b")
    plt.bar("5", true_list_colour_score.count(5), color="b")
    plt.bar("6",true_list_colour_score.count(6), color="b")
    plt.xlabel("Color score")
    plt.ylabel("Amount")
    plt.title("Color score diagram")
    plt.legend()
    plt.savefig("Color score diagram melanoma")
    plt.show()
    
    plt.bar("0", false_list_colour_score.count(0), label="non-melanoma", color="r")
    plt.bar("1", false_list_colour_score.count(1), color="r")
    plt.bar("2", false_list_colour_score.count(2), color="r")
    plt.bar("3", false_list_colour_score.count(3), color="r")
    plt.bar("4", false_list_colour_score.count(4), color="r")
    plt.bar("5", false_list_colour_score.count(5), color="r")
    plt.bar("6", false_list_colour_score.count(6), color="r")
    plt.xlabel("Color score")
    plt.ylabel("Amount")
    plt.title("Color score diagram")
    plt.legend()
    plt.savefig("Color score diagram non-melanoma")
    plt.show()
    
    plt.plot(false_list_area,false_list_colour_cluster,'b.', label="non-melanoma")
    plt.plot(true_list_area,true_list_colour_cluster,'r.', label="melanoma")
    plt.xlabel("Area")
    plt.ylabel("Cluster score")
    plt.title("Cluster score-Area diagram")
    plt.legend()
    plt.savefig("Cluster score-Area diagram")
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.boxplot([true_list_colour_cluster,false_list_colour_cluster])
    # ax.set_xticklabels( ['melanoma','non-melanoma'] )
     
make_plots()
