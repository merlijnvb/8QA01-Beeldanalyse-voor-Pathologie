import cv2
import numpy as np
import os

"""READ IMAGES AND CONVERT THEM"""
def img_conversion(mask_file,lesion_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
 
    lesion = cv2.imread(lesion_file)
    lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
    
    #REMOVE OLD IMAGES AND REPLACE THEM WITH NEW ONES.
    os.remove("original_lesion.png")
    cv2.imwrite("original_lesion.png", lesion)
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    angle = cv2.fitEllipse(contours[0])[2] - 90
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))

    os.remove("rotated_mask.png")
    os.remove("rotated_lesion.png")
    cv2.imwrite("rotated_mask.png", mask)
    cv2.imwrite("rotated_lesion.png", lesion)

    lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
    
    return (mask,lesion)

"""EVALUATE LESIONS"""
def border_evaluation(mask):       
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 3)
    
    #REMOVE OLD IMAGES AND REPLACE THEM WITH NEW ONES.
    os.remove("border.png")
    cv2.imwrite("border.png", border)
    
def symmetry_evaluation(mask):
    height, width = mask.shape[:2]
    moment = cv2.moments(mask)
    
    centre_blob_x = int(moment["m10"] / moment["m00"])
    centre_blob_y = int(moment["m01"] / moment["m00"])

    superior = mask[0:centre_blob_y, 0:width]
    inferior = mask[centre_blob_y:height, 0:width]
    inferior = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_blob_x]
    left = cv2.flip(left, 1)
    right = mask[0:height, centre_blob_x:width]
    
    if superior.shape[0] > inferior.shape[0]:
        inferior = cv2.copyMakeBorder(inferior, superior.shape[0]-inferior.shape[0], None, None, None, 0, None, None)                     
        horizontal_result = superior - inferior
        
    if superior.shape[0] < inferior.shape[0]:
        superior = cv2.copyMakeBorder(superior, inferior.shape[0]-superior.shape[0], None, None, None, 0, None, None)  
        horizontal_result = inferior - superior
    
    if left.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left.shape[1]-right.shape[1], 0, None, None)
        vertical_result = left - right
        
    if left.shape[1] < right.shape[1]:
        left = cv2.copyMakeBorder(left, None, None, None, right.shape[1]-left.shape[1], 0, None, None)
        vertical_result = right - left
        
    if left.shape[1] == right.shape[1]:  
        vertical_result = right - left
    
    if superior.shape[0] == inferior.shape[0]:
        horizontal_result = superior - inferior
    
    #REMOVE OLD IMAGES AND REPLACE THEM WITH NEW ONES.
    os.remove("inferior_side.png")
    os.remove("superior_side.png")
    os.remove("left_side.png")
    os.remove("right_side.png")
    os.remove("horizontal_overlapse.png")
    os.remove("vertical_overlapse.png")
    cv2.imwrite("inferior_side.png", inferior)
    cv2.imwrite("superior_side.png", superior)
    cv2.imwrite("left_side.png", left)
    cv2.imwrite("right_side.png", right)
    cv2.imwrite("horizontal_overlapse.png", horizontal_result)
    cv2.imwrite("vertical_overlapse.png", vertical_result)

"""CALL EVERY FUNCTION 1 TIME TO SHOW RESULTS"""
def return_results(mask_file, lesion_file):
    mask, lesion = img_conversion(mask_file, lesion_file)
    
    symmetry_evaluation(mask)
    border_evaluation(mask)
    
#INPUT FILE NAMES FOR WICH YOU WANT TO GET IMAGES AS REULTS
mask_file = "ISIC-TrainValTest\masks\ISIC_0013319_segmentation.png"
lesion_file = "ISIC-TrainValTest\lesions\ISIC_0013319.jpg"
return_results(mask_file, lesion_file)
