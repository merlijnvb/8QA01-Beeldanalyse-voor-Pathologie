# GROUP 5

Skin cancer accounts for over 40% of all cancer cases. By detecting skin cancer early, by noticing abnormal changes in birth marks on the skin, survival can be dramatically improved. However, people might hesitate to consult a doctor for any new change that they see. Using a smartphone app as a first check could help people   determine   whether   a   change   is   really   nothing,   or   whether   they   really   should   make   an appointment.


**The central question:** _How accurate can an algorithm recognize a skin lesion as melanoma, based on its color, asymmetry, diameter and border characteristics from a photo?_


### Approach
Moles can be evaluated by using the [ABCDE's](https://www.skincancer.org/skin-cancer-information/melanoma/melanoma-warning-signs-and-images/). This evaluation method can be implemented in an algorithm to get a more objective and more precise evaluation.



### (A) asymmetry
Asymmetry can be evaluated by the following formula: `Asym = (Area_overlapse_x_y) / Area_blob`.

```markdown
import cv2
import numpy as np

height, width = mask.shape[:2]
moment = cv2.moments(mask)

centre_blob_x = int(moment["m10"] / moment["m00"])
centre_blob_y = int(moment["m01"] / moment["m00"])

segmet_one = mask[x1:y1 , x2:y2]
segment_two = mask[x2:y2 , x3:y3]
segment_two = cv2.flip(segment_two, 0)

overlapse = np.sum((segment_one - segment_two) == 255)
area = np.sum(mask == 255)

Asym = overlapse / area
```

This gives the following reult: [Image](https://github.com/merlijnvb/8QA01-Beeldanalyse-voor-Pathologie/blob/main/Merlijn/resutls/left_side.png)

### (B) border
Border can be evaluated by the following formula: `Bor = Area_border / Area_blob`.

```
border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 3)

Bor = np.sum(border == 255)
```

This gives the following reult: [Image](https://github.com/merlijnvb/8QA01-Beeldanalyse-voor-Pathologie/blob/main/Merlijn/resutls/border.png)

### (C) colour
Border can be evaluated by the following formula: `if Area_colour > 5% * Area_blob: Col += 1`.

```
mask_colour = cv2.inRange(lesion, lower_end, higher_end)

if (np.sum(mask_colour == 255) / area) >= 0.05:
  colour_score += 1
```

This gives the following reult: [Image](https://github.com/merlijnvb/8QA01-Beeldanalyse-voor-Pathologie/blob/main/Merlijn/resutls/coloured_mask_0.png)

### (D) diameter
Diameter can be evaluated by the following formula: `Dia = diameter / Area_blob`.

```
import math

diameter = border / math.pi
Dia = diameter / area
```
