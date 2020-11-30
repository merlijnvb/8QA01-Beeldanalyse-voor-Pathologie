import cv2

def diameter_evaluation(mask):
    moment = cv2.moments(mask)
    
    diameter_score = moment["m10"]/moment["m01"]
    
    return diameter_score