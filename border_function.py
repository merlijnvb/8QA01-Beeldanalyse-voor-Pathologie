import cv2
import numpy as np
import math

def border_evaluation(mask):
    height, width = mask.shape[:2]
    dim = (width-1, height-1)  

    resized = cv2.resize(mask, dim)
    resized = cv2.copyMakeBorder(resized, 1, None, 1, None, 0, None, None)
    border = mask - resized
    
    length_border = np.sum(border == 255)
    area_mask = np.sum(mask == 255)

    border_score = (length_border**2) / (4*math.pi*area_mask)
    
    return (border_score, border)