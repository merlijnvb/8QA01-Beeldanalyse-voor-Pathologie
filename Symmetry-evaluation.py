import cv2
import numpy as np
import matplotlib.pyplot as plt

# image flipping
img = cv2.imread('melanoma.png')

h = img.shape[0]
w = img.shape[1]

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

#img[ y1:y2   ,    x1:x2    ]
boven = img[0:cY, 0:w]
onder = img[cY:h, 0:w]
onder_flipped = cv2.flip(onder, 0)
links = img[0:h, 0:cX]
links_flipped = cv2.flip(links, 1)  
rechts = img[0:h, cX:w]
    
if boven.shape[0] > onder_flipped.shape[0]:
    onder_flipped = cv2.copyMakeBorder(onder_flipped, boven.shape[0]-onder_flipped.shape[0], None, None, None, 0, None, None)
      
    mask_boven = cv2.inRange(boven, (1,1,1), (255,255,255))
    mask_onder = cv2.inRange(onder_flipped, (1,1,1), (255,255,255))
        
    resultaat = mask_boven - mask_onder
    
if boven.shape[0] < onder_flipped.shape[0]:
    boven = cv2.copyMakeBorder(boven, onder_flipped.shape[0]-boven.shape[0], None, None, None, 0, None, None)
        
    mask_boven = cv2.inRange(boven, (1,1,1), (255,255,255))
    mask_onder = cv2.inRange(onder_flipped, (1,1,1), (255,255,255))
        
    resultaat_horizontal = mask_onder - mask_boven

if links_flipped.shape[1] > rechts.shape[1]:
    rechts = cv2.copyMakeBorder(rechts, None, None, None, links_flipped.shape[1]-rechts.shape[1], 0, None, None)
        
    mask_rechts = cv2.inRange(rechts, (1,1,1), (255,255,255))
    mask_links = cv2.inRange(links_flipped, (1,1,1), (255,255,255))
        
    resultaat_vertical = mask_links - mask_rechts
    
if links_flipped.shape[1] < rechts.shape[1]:
    links_flipped = cv2.copyMakeBorder(links_flipped, None, None, None, rechts.shape[1]-links_flipped.shape[1], 0, None, None)
        
    mask_rechts = cv2.inRange(rechts, (1,1,1), (255,255,255))
    mask_links = cv2.inRange(links_flipped, (1,1,1), (255,255,255))
        
    resultaat_vertical = mask_rechts - mask_links
    
fig, axs = plt.subplots(3,2)
axs[0, 0].set_title("Horizontale doorsnede")
axs[0, 1].set_title("Verticale doorsnede")

axs[0,0].imshow(mask_boven, cmap="gray")
axs[1,0].imshow(mask_onder, cmap="gray")
axs[2,0].imshow(resultaat_horizontal, cmap="gray")

axs[0,1].imshow(mask_rechts, cmap="gray")
axs[1,1].imshow(mask_links, cmap="gray")
axs[2,1].imshow(resultaat_vertical, cmap="gray")

pix_melanoma = np.sum(img > 0)
pix_diff_vertical = np.sum(resultaat_vertical > 0)
pix_diff_horizontal = np.sum(resultaat_horizontal > 0)

quotient_vertical = pix_diff_vertical / pix_melanoma
quotient_horizontal = pix_diff_horizontal / pix_melanoma

print("Je verticale symmetrie score =", quotient_vertical)
print("Je horizontale symmetrie score =", quotient_horizontal)
