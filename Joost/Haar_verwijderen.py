# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:58:01 2020

@author: 20203167
"""

import cv2
def haarverwijderen(foto):
    src = cv2.imread(foto)
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    thresh2 = cv2.dilate(thresh2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cv2.imwrite('C:/Users/20203167/Documents//thresh2.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    dst = cv2.inpaint(src,thresh2, 3, cv2.INPAINT_TELEA)
    cv2.imshow('dst', dst)
    # cv2.imwrite('C:/Users/20203167/Documents//InPainted_sample2.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return dst
    

 
haarverwijderen('C:/Users/20203167/Documents/Bekijken Plaatjes/Onze data/ISIC_0014503.jpg')