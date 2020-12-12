#https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html
#https://answers.opencv.org/question/204175/how-to-get-boundry-and-center-information-of-a-mask/

import math

file = open("results.txt")
file_read = file.readlines()
file.close()

value_data = []

for lines in file_read[1:]:
    lines = lines[:-1]
    lines = tuple(lines.split(","))
    value_data.append(lines)

list_borders = []
list_areas = []
list_symmetry_vertical = []
list_symmetry_horizontal = []
list_colour_scores = []
list_diameters = []



for tupl in value_data:
    
    border = int(tupl[1])
    area = int(tupl[2])
    symmetry_vertical = int(tupl[3])
    symmetry_horizontal = int(tupl[4])
    colour_score = int(tupl[5])
    
    diameter = border / math.pi

    list_borders.append(border)
    list_areas.append(area)
    list_symmetry_vertical.append(symmetry_vertical)
    list_symmetry_horizontal.append(symmetry_horizontal)
    list_colour_scores.append(colour_score)
    list_diameters.append(diameter)
    