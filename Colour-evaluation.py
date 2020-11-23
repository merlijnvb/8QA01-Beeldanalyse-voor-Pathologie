import cv2
import numpy as np
import matplotlib.pyplot as plt

leasie = cv2.imread("Melanoma.png")
color_score = 0
leasie = cv2.cvtColor(leasie, cv2.COLOR_BGR2RGB)

# intervals
light_brown_higher_range = (109, 60, 46)
light_brown_lower_range = (92, 64, 51)

dark_brown_higher_range = (92, 64, 51)
dark_brown_lower_range = (43, 30, 24)

white_higher_range = (255, 255, 255)
white_lower_range = (217, 217, 217)

red_higher_range = (255, 77, 77)
red_lower_range = (154, 0, 0)

blue_grey_higher_range = (144, 168, 180)
blue_grey_lower_range = (69, 91, 102)

black_higher_range = (38, 38, 38)
black_lower_range = (0, 0, 0)

# mask of colour
mask_light_brown = cv2.inRange(leasie, light_brown_lower_range, light_brown_higher_range)
mask_dark_brown = cv2.inRange(leasie, dark_brown_lower_range, dark_brown_higher_range)
mask_white = cv2.inRange(leasie, white_lower_range, white_higher_range)
mask_red = cv2.inRange(leasie, red_higher_range, red_higher_range)
mask_blue_grey = cv2.inRange(leasie, blue_grey_lower_range, blue_grey_higher_range)
mask_black = cv2.inRange(leasie, black_lower_range, black_higher_range)

# show images

fig, ax = plt.subplots(3, 2)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
fig.suptitle('Masks of colours', fontsize=14)

ax[0, 0].imshow(mask_light_brown, cmap="gray")
ax[0, 1].imshow(mask_dark_brown, cmap="gray")
ax[1, 0].imshow(mask_white, cmap="gray")
ax[1, 1].imshow(mask_black, cmap="gray")
ax[2, 0].imshow(mask_red, cmap="gray")
ax[2, 1].imshow(mask_blue_grey, cmap="gray")

ax[0, 0].set_title("light brown")
ax[0, 1].set_title("dark brown")
ax[1, 0].set_title("white")
ax[1, 1].set_title("black")
ax[2, 0].set_title("red")
ax[2, 1].set_title("blue grey")


# area colours
pix_melanoma = np.sum(leasie > 0)
pix_negative = np.sum(leasie == 255)
pix_light_brown = np.sum(mask_light_brown == 255)
pix_dark_brown = np.sum(mask_dark_brown == 255)
pix_white = np.sum(mask_white == 255)
pix_red = np.sum(mask_red == 255)
pix_blue_gray = np.sum(mask_blue_grey == 255)
pix_black = np.sum(mask_black == 255) - pix_negative

#counting system
for i in [pix_dark_brown, pix_light_brown, pix_white, pix_red, pix_blue_gray, pix_black]:
    if (i/pix_melanoma) >= 0.05:
        color_score += 1

print("Je 'C' score is: ", color_score)

if color_score == 0:
    #consequentie
    print()

if color_score == 1:
    #consequentie 
    print()

if color_score == 2:
    #consequentie 
    print()

if color_score == 3:
    #consequentie 
    print()

if color_score == 4:
    #consequentie 
    print()

if color_score == 5:
    #consequentie         
    print()

if color_score == 6:
    #consequentie     
    print()
