import cv2
import numpy as np

def colour_evaluation(lesion, mask):
    color_score = 0
    
    # intervals
    light_brown_higher_range = (249, 193, 160)
    light_brown_lower_range = (55, 24, 22)
    
    dark_brown_higher_range = (55, 24, 22)
    dark_brown_lower_range = (36, 15, 15)
    
    white_higher_range = (255, 255, 255)
    white_lower_range = (217, 217, 217)
    
    red_higher_range = (255, 77, 77)
    red_lower_range = (154, 0, 0)
    
    blue_grey_higher_range = (144, 168, 180)
    blue_grey_lower_range = (69, 91, 102)
    
    black_higher_range = (38, 38, 38)
    black_lower_range = (0, 0, 0)
    
    # mask of colour
    mask_light_brown = cv2.inRange(lesion, light_brown_lower_range, light_brown_higher_range)
    mask_dark_brown = cv2.inRange(lesion, dark_brown_lower_range, dark_brown_higher_range)
    mask_white = cv2.inRange(lesion, white_lower_range, white_higher_range)
    mask_red = cv2.inRange(lesion, red_lower_range, red_higher_range)
    mask_blue_grey = cv2.inRange(lesion, blue_grey_lower_range, blue_grey_higher_range)
    mask_black = cv2.inRange(lesion, black_lower_range, black_higher_range)
    
    # area colours
    pix_melanoma = np.sum(mask == 255)
    pix_light_brown = np.sum(mask_light_brown == 255)
    pix_dark_brown = np.sum(mask_dark_brown == 255)
    pix_white = np.sum(mask_white == 255)
    pix_red = np.sum(mask_red == 255)
    pix_blue_gray = np.sum(mask_blue_grey == 255)
    pix_black = np.sum(mask_black == 255) - ((lesion.shape[0] * lesion.shape[1]) - pix_melanoma)
    
    #counting system
    for i in [pix_dark_brown, pix_light_brown, pix_white, pix_red, pix_blue_gray, pix_black]:
        if (i/pix_melanoma) >= 0.05:
            color_score += 1
            
    return (color_score,mask_light_brown,mask_dark_brown,mask_white,mask_red,mask_blue_grey,mask_black)