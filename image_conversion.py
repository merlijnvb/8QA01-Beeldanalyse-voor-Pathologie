import cv2  
 
def image_rotation(lesion, mask):
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    has_ellipse = len(contours) > 0
    
    if has_ellipse:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        angle = ellipse[2] - 90
        x, y = ellipse[1]
    
    
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, moment, (width, height))
    rotated_mask_scaled = cv2.copyMakeBorder(rotated_mask, 25, 25, 25, 25, 0, None, None)
    
    rotated_image = cv2.warpAffine(lesion, moment, (width, height))
    rotated_image_scaled = cv2.copyMakeBorder(rotated_image, 25, 25, 25, 25, 0, None, None)
    
    return (rotated_mask_scaled, rotated_image_scaled)