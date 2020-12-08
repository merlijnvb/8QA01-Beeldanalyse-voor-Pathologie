import cv2
import os
import symmetry_function as sym
import image_conversion as conv
import colour_function as col
import border_function as bor
import diameter_function as dia
from tqdm import tqdm

def read_files():
    lesion_img_list = []
    lesion_mask_list = []
    lesion_list = []
    
    for lesion in os.listdir("data"):
        if lesion.endswith(".jpg"):
            lesion_img_list.append(lesion)
            
    for mask in os.listdir("data_masks"):
        if mask.endswith(".png"):
            lesion_mask_list.append(mask)
            
    for i in range(len(lesion_img_list)):
        img = lesion_img_list[i]
        img_mask = lesion_mask_list[i]
        lesion_list.append((img,img_mask))
        
    return lesion_list

def image_evaluator(lesion,mask):
    #lesion_mask_rotated = conv.image_rotation(lesion,mask)[0]
    #lesion_image_rotated = conv.image_rotation(lesion,mask)[1]
    
    lesion_mask_rotated = mask
    lesion_image_rotated = lesion
        
    sym_score = sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[0] + sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[1]
    bor_score = bor.border_evaluation(lesion_mask_rotated)[0]
    col_score = col.colour_evaluation(lesion_image_rotated, lesion_mask_rotated)[0]
    dia_score = dia.diameter_evaluation(lesion_mask_rotated)
    
    return (sym_score,bor_score,col_score,dia_score)

def return_results():
    data = read_files()
    read_file = open("results.txt", "w")
    
    results_images = []
    
    for nr in tqdm(range(len(data))):
        ID = nr
        lesion = cv2.imread("data\{}".format(data[nr][0]))
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread("data_masks\{}".format(data[nr][1]), cv2.IMREAD_GRAYSCALE)
        lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
                
        sym_score = image_evaluator(lesion,mask)[0]
        bor_score = image_evaluator(lesion,mask)[1]
        col_score = image_evaluator(lesion,mask)[2]
        dia_score = image_evaluator(lesion,mask)[3]
        
        results_images.append((ID,sym_score,bor_score,col_score,dia_score))
       

    for result in results_images:
        line = "{0:d},{1:f},{2:f},{3:d},{4:f}\n".format(result[0],result[1],result[2],result[3],result[4])
        read_file.write(line)
        
    read_file.close()  
        
return_results() 

