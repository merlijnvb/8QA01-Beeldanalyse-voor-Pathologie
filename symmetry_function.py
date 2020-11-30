import cv2
import numpy as np

def symmetry_evaluation(lesion, mask):
    
    height, width = lesion.shape[:2]
      
    # calculate moments of binary image
    moment = cv2.moments(mask)
    
    # calculate x,y coordinate of center
    centre_x = int(moment["m10"] / moment["m00"])
    centre_y = int(moment["m01"] / moment["m00"])
    
    #lesion[ y1:y2   ,    x1:x2    ]
    superior = mask[0:centre_y, 0:width]
    inferior = mask[centre_y:height, 0:width]
    inferior_flipped = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_x]
    left_flipped = cv2.flip(left, 1)
    right = mask[0:height, centre_x:width]
    
        
    if superior.shape[0] > inferior_flipped.shape[0]:
        inferior_flipped = cv2.copyMakeBorder(inferior_flipped, superior.shape[0]-inferior_flipped.shape[0], None, None, None, 0, None, None)
                     
        resultaat_horizontal = superior - inferior_flipped
        
    if superior.shape[0] < inferior_flipped.shape[0]:
        superior = cv2.copyMakeBorder(superior, inferior_flipped.shape[0]-superior.shape[0], None, None, None, 0, None, None)
            
        resultaat_horizontal = superior - inferior_flipped
    
    if superior.shape[0] < inferior_flipped.shape[0]:
            
        resultaat_horizontal = inferior_flipped - superior
    
    if left_flipped.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left_flipped.shape[1]-right.shape[1], 0, None, None)
            
        resultaat_vertical = left_flipped - right
        
    if left_flipped.shape[1] < right.shape[1]:
        left_flipped = cv2.copyMakeBorder(left_flipped, None, None, None, right.shape[1]-left_flipped.shape[1], 0, None, None)
            
        resultaat_vertical = right - left_flipped
        
    if left_flipped.shape[1] == right.shape[1]:
            
        resultaat_vertical = right - left_flipped
    
    pix_melanoma = np.sum(lesion > 0)
    pix_diff_vertical = np.sum(resultaat_vertical > 0)
    pix_diff_horizontal = np.sum(resultaat_horizontal > 0)
    
    quotient_vertical = pix_diff_vertical / pix_melanoma
    quotient_horizontal = pix_diff_horizontal / pix_melanoma
    
    return (quotient_vertical,quotient_horizontal,right,left_flipped,superior,inferior_flipped,resultaat_horizontal,resultaat_vertical)