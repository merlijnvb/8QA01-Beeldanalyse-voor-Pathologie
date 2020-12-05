import cv2
import image_conversion as conv

import symmetry_function as sym
import border_function as bor
import colour_function as col
import diameter_function as dia

lesion = cv2.imread("Melanoma.png") #can be any cut out of lesion
lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(lesion, cv2.COLOR_RGB2GRAY)
ret,mask = cv2.threshold(gray_image,0,255,0)

lesion_mask_rotated = conv.image_rotation(lesion,mask)[0]
lesion_image_rotated = conv.image_rotation(lesion,mask)[1]

print("(A) symmetry score =",sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[0] + sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[1])
print("(B) border score =", bor.border_evaluation(lesion_mask_rotated)[0])
print("(C) color score =",col.colour_evaluation(lesion_image_rotated, lesion_mask_rotated))
print("(D) diameter score =", dia.diameter_evaluation(lesion_mask_rotated))
