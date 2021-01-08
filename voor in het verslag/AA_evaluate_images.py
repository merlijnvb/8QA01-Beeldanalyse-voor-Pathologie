import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


"""
    GET ALL FILENAMES WITH LABELS.
"""
def read_files():
    csv = open("labels.csv")
    csv_read = csv.readlines()
    csv.close()
    
    lesion_list = []
    
    #PICK THE IMAGENAMES FROM THE LABELFILE AND SAVE THEM IN A IMAGES LIST.
    for lines in csv_read[1:]:
        ID = lines.rstrip().split(",")[0]
        label = lines.rstrip().split(",")[1]
        lesion_list.append(("{}_segmentation.png".format(ID),"{}.jpg".format(ID),label))
    
    return lesion_list


"""
    READ IMAGES AND CONVERT THEM.
"""
def img_conversion(mask_file,lesion_file):
    #OPEN IMAGES IN OPENCV.
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
 
    lesion = cv2.imread(lesion_file)
    lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
    lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
    
    #GET DIMENSIONS
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    #TRY TO FIND ANGLE.
    try:
        angle = cv2.fitEllipse(contours[0])[2] - 90
        
    except:
        angle = 0

    #MAKE MOMENT OF IMAGE WITH THE CHANGED ANGLE.
    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    #ROTATE THE IMAGES.
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))
    
    area = np.sum(mask == 255)
    
    return (mask, lesion, area)


"""
    EVALUATE LESIONS.
"""
def border_evaluation(mask):
    #MAKE EMPTY/BLACK CANVAS/IMAGE.
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    #DRAW THE FOUND BORDER ON THE EMPTY CANVAS.
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 3)
    
    #GET BORDER LENGTH.
    length_border = np.sum(border == 255)

    return length_border

def color_cluster_evaluation(lesion, clusters = 5):
    #GET DIMENSIONS
    w, h, d = lesion.shape[:]
    image_array = np.reshape(lesion, (w * h, d)) 
    image_array_sample = shuffle(image_array, random_state=0)[:10000] 
    
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_array_sample)
    centroids = kmeans.cluster_centers_
    
    D = cdist(centroids, centroids, metric='seuclidean')
    total = 0
    
    #GET TOTAL DISTANCE.
    for i in range(clusters):
        for j in range(clusters):
            if i<j:
                total = total + D[i, j]

    mean_dinstance = total/((clusters*(clusters-1))/2)

    return mean_dinstance
    
def symmetry_evaluation(mask):
    #GET DIMENSIONS.
    height, width = mask.shape[:2]
    moment = cv2.moments(mask)
    
    #FIND THE CENTRE OF THE BLOB. ==> "m10": total x pixels; "m01" total y pixels; "m00" total pixels whole blob.
    centre_blob_x = int(moment["m10"] / moment["m00"])
    centre_blob_y = int(moment["m01"] / moment["m00"])

    #CUT IMAGES IN CENTRE OF THE x AND y AXIS.
    superior = mask[0:centre_blob_y, 0:width]
    inferior = mask[centre_blob_y:height, 0:width]
    inferior = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_blob_x]
    left = cv2.flip(left, 1)
    right = mask[0:height, centre_blob_x:width]
    
    #ADD EMPTY SPACE TO THE SMALLER IMAGE SO BOTH IMAGES CAN BE SUBTRACTED FROM EACHOTHER.
    if superior.shape[0] > inferior.shape[0]:
        inferior = cv2.copyMakeBorder(inferior, superior.shape[0]-inferior.shape[0], None, None, None, 0, None, None)                     
        horizontal_result = superior - inferior
        
    if superior.shape[0] < inferior.shape[0]:
        superior = cv2.copyMakeBorder(superior, inferior.shape[0]-superior.shape[0], None, None, None, 0, None, None)  
        horizontal_result = inferior - superior
    
    if left.shape[1] > right.shape[1]:
        right = cv2.copyMakeBorder(right, None, None, None, left.shape[1]-right.shape[1], 0, None, None)
        vertical_result = left - right
        
    if left.shape[1] < right.shape[1]:
        left = cv2.copyMakeBorder(left, None, None, None, right.shape[1]-left.shape[1], 0, None, None)
        vertical_result = right - left
        
    if left.shape[1] == right.shape[1]:  
        vertical_result = right - left
    
    if superior.shape[0] == inferior.shape[0]:
        horizontal_result = superior - inferior
        
    #GET OVERLAPSE AREA.
    horizontal_result = np.sum(horizontal_result == 255)
    vertical_result = np.sum(vertical_result == 255)
    
    return (horizontal_result,vertical_result)


"""
    RETURN RESULTS OF LESIONS IN CSV FILE.
"""
def return_results():
    #OPEN DOCUMENT TO INSERT RESULTS.
    document = open("results.csv", "w")
    data = read_files()
    
    #GET DATA FROM THE FILE AND WRITE THE RESULTS FILE.
    for fileset in tqdm(data):
        index = fileset[1][:-4]
        lesion_label = fileset[2]
        
        #WRITE FIRST LINE.
        if index == data[0][1][:-4]:
            document.write("index,lng_bor,area,hor_overl,vrt_overl,cluster_clr_score,melanoma_label\n")
        
        #GET FILE ADRES.
        mask_file = "ISIC-TrainValTest\masks\{}".format(fileset[0])
        lesion_file = "ISIC-TrainValTest\lesions\{}".format(fileset[1])
        
        #GET DATA.
        mask, lesion, area = img_conversion(mask_file, lesion_file)
        
        lng_bor = border_evaluation(mask)
        hor_overl, vrt_overl = symmetry_evaluation(mask)
        cluster_clr_score = color_cluster_evaluation(lesion)
        
        #FORMAT DATA IN LINES FOR RESULTS FILE.
        line = "{0:s},{1:d},{2:d},{3:d},{4:d},{5:f},{6:s}\n".format(index,lng_bor,area,hor_overl,vrt_overl,cluster_clr_score,lesion_label)
        document.write(line)
    document.close()

return_results()
