import cv2
import numpy as np
import matplotlib.pyplot as plt

"""READ IMAGES AND CONVERT THEM"""
def img_conversion(mask_file,lesion_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
 
    lesion = cv2.imread(lesion_file)
    lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
    lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
    
    plt.imshow(mask)
    plt.suptitle("Original Mask")
    plt.show()
    plt.imshow(lesion)
    plt.suptitle("Original Lesion")
    plt.show()
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    angle = cv2.fitEllipse(contours[0])[2] - 90
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))
    
    plt.imshow(mask)
    plt.suptitle("Rotated Mask")
    plt.show()
    plt.suptitle("Rotated Lesion")
    plt.imshow(lesion)
    plt.show()    
    
    return (mask, lesion)

"""EVALUATE LESIONS"""
def border_evaluation(mask):       
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 3)
    
    length_border = np.sum(border == 255)
    
    plt.imshow(border)
    plt.suptitle("Border")
    plt.show()   
    
    return length_border
    
def colour_evaluation(mask, lesion):  
    mask_inv = 255 - mask
    area = np.sum(mask == 255)
    colour_score = 0
    
    colours = {'light brown low':(255*0.588, 255*0.2, 255*0),
              'light brown high':(255*0.94, 255*0.588, 255*392),
              'dark brown low':(255*0.243, 255*0, 255*0),
              'dark borwn high':(255*56, 255*0.392, 255*392),
              'white low':(255*0.8, 255*0.8, 255*0.8),
              'white high':(255, 255, 255),
              'red low':(255*0.588, 255*0, 255*0),
              'red high':(255, 255*0.19, 255*0.19),
              'blue gray low':(255*0, 255*0.392, 255*0.490),
              'blue gray high':(255*0.588, 255*0.588, 255*0.588),
              'black low':(255*0, 255*0, 255*0),
              'black high':(255*0.243, 255*0.243, 255*0.243)}
    
    for i in range(0,len(colours),2):
        mask_colour = cv2.inRange(lesion, colours.get(list(colours.keys())[i]), colours.get(list(colours.keys())[i+1]))
    
        if list(colours.keys())[i] == list(colours.keys())[-2] and list(colours.keys())[i+1] == list(colours.keys())[-1]:
            mask_colour = mask_colour - mask_inv
        
        plt.imshow(mask_colour)
        plt.suptitle("colour range: {}".format([i]))
        plt.show() 
        
        if (np.sum(mask_colour == 255) / area) >= 0.05:
            colour_score += 1
        
    return (area,colour_score)
    
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
        
    plt.imshow(horizontal_result)
    plt.suptitle("Horizontal overlapse")
    plt.show()
    plt.suptitle("Vertical overlapse")
    plt.imshow(vertical_result)
    plt.show() 
        
    horizontal_result = np.sum(horizontal_result == 255)
    vertical_result = np.sum(vertical_result == 255)
    
    return (horizontal_result,vertical_result)

"""CALL EVERY FUNCTION 1 TIME TO SHOW RESULTS"""
def return_results(mask_file, lesion_file):
    mask, lesion = img_conversion(mask_file, lesion_file)   
    
    symmetry_evaluation(mask)
    border_evaluation(mask)
    colour_evaluation(mask, lesion)
    
#INPUT FILE NAMES FOR WICH YOU WANT TO GET IMAGES AS REULTS
mask_file = "data_masks\ISIC_0013319_segmentation.png"
lesion_file = "data\ISIC_0013319.jpg"
return_results(mask_file, lesion_file)
