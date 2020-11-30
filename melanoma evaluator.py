import cv2
import symmetry_function as sym
import image_conversion as conv
import colour_function as col

lesion = cv2.imread("Melanoma.png") #can be any cut out of lesion
lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(lesion, cv2.COLOR_RGB2GRAY)
ret,mask = cv2.threshold(gray_image,0,255,0)

lesion_mask_rotated = conv.image_rotation(lesion,mask)[0]
lesion_image_rotated = conv.image_rotation(lesion,mask)[1]

print("colour score = ",col.colour_evaluation(lesion_image_rotated, lesion_mask_rotated)[0])
    
print("symmetry score = ",sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[0] + sym.symmetry_evaluation(lesion_image_rotated, lesion_mask_rotated)[1])
