import cv2
import numpy as np
import matplotlib.pyplot as plt

def colour_evaluation(lesion, mask):
    mask_inv = 255 - mask
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
        
        result = cv2.bitwise_and(lesion, lesion, mask=mask_colour)
        
        if (np.sum(mask_colour == 255) / np.sum(mask == 255)) >= 0.05:    
            colour_score += 1
        
        
        fig, axs = plt.subplots(2)
        fig.suptitle("{} <-> {}".format(list(colours.keys())[i],list(colours.keys())[i+1]))
        axs[0].imshow(mask_colour, cmap="gray")
        axs[1].imshow(result)
        plt.show()
        
    return colour_score
