"""GET DATA FROM PHOTO's"""

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


"""
    GET ALL FILENAMES WITH LABELS
"""
def read_files():
    csv = open("labels.csv")
    csv_read = csv.readlines()
    csv.close()
    
    lesion_list = []
    
    for lines in csv_read[1:]:
        ID = lines.rstrip().split(",")[0]
        label = lines.rstrip().split(",")[1]
        lesion_list.append(("{}_segmentation.png".format(ID),"{}.jpg".format(ID),label))
    
    return lesion_list


"""
    READ IMAGES AND CONVERT THEM
"""
def img_conversion(mask_file,lesion_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
 
    lesion = cv2.imread(lesion_file)
    lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)
    lesion = cv2.bitwise_and(lesion, lesion, mask=mask)
    
    height, width = mask.shape[:2]
    centre = (width // 2, height // 2)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    try:
        angle = cv2.fitEllipse(contours[0])[2] - 90
        
    except:
        angle = 0

    moment = cv2.getRotationMatrix2D(centre, angle, 1.0)
    
    mask = cv2.warpAffine(mask, moment, (width, height))
    lesion = cv2.warpAffine(lesion, moment, (width, height))
    
    area = np.sum(mask == 255)
    
    return (mask, lesion, area)


"""
    EVALUATE LESIONS
"""
def border_evaluation(mask):       
    border_blanc = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]    
    contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        
    border = cv2.drawContours(border_blanc,contours, 0, (255, 255, 255), 3)
    
    length_border = np.sum(border == 255)

    return length_border

def color_cluster_evaluation(lesion, clusters = 5):    
    w, h, d = lesion.shape[:]
    image_array = np.reshape(lesion, (w * h, d)) 
    image_array_sample = shuffle(image_array, random_state=0)[:10000] 
    
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(image_array_sample)
    centroids = kmeans.cluster_centers_
    
    D = cdist(centroids, centroids, metric='seuclidean')
    total = 0
    
    for i in range(clusters):
        for j in range(clusters):
            if i<j:
                total = total + D[i, j]

    mean_dinstance = total/((clusters*(clusters-1))/2)

    return mean_dinstance
    
def symmetry_evaluation(mask):
    height, width = mask.shape[:2]
    moment = cv2.moments(mask)
    
    centre_blob_x = int(moment["m10"] / moment["m00"])
    centre_blob_y = int(moment["m01"] / moment["m00"])

    superior = mask[0:centre_blob_y, 0:width]
    inferior = mask[centre_blob_y:height, 0:width]
    inferior = cv2.flip(inferior, 0)
    
    left = mask[0:height, 0:centre_blob_x]
    left = cv2.flip(left, 1)
    right = mask[0:height, centre_blob_x:width]
    
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
        
    horizontal_result = np.sum(horizontal_result == 255)
    vertical_result = np.sum(vertical_result == 255)
    
    return (horizontal_result,vertical_result)


"""
    RETURN RESULTS OF LESIONS IN CSV FILE
"""
def return_results():
    #OPEN DOCUMENT TO INSERT RESULTS
    document = open("results.csv", "w")
    data = read_files()
    
    for fileset in tqdm(data):
        index = fileset[1][:-4]
        lesion_label = fileset[2]
        
        if index == data[0][1][:-4]:
            document.write("index,lng_bor,area,hor_overl,vrt_overl,cluster_clr_score,melanoma_label\n")
        
        mask_file = "ISIC-TrainValTest\masks\{}".format(fileset[0])
        lesion_file = "ISIC-TrainValTest\lesions\{}".format(fileset[1])
        
        mask, lesion, area = img_conversion(mask_file, lesion_file)
        
        lng_bor = border_evaluation(mask)
        hor_overl, vrt_overl = symmetry_evaluation(mask)
        cluster_clr_score = color_cluster_evaluation(lesion)
        
        line = "{0:s},{1:d},{2:d},{3:d},{4:d},{5:f},{6:s}\n".format(index,lng_bor,area,hor_overl,vrt_overl,cluster_clr_score,lesion_label)
        document.write(line)
    document.close()

return_results()

"""GET BEST CLASSIFIER"""

import math
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid

def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_groups = []
    value_data = []
       
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines[:-1])
        control_groups.append(lines[-1])
    
    return value_data, control_groups

def extract_info(value_data, control_groups):
    
    list_id = []
    list_data = []
    
    for tupl in value_data:
        ID = tupl[0]
        border = int(tupl[1])
        area = int(tupl[2])
        symmetry_overlapse = int(tupl[3]) + int(tupl[4])
        colour_cluster_score = float(tupl[5])
                
        border_score = (border**2) / (area*math.pi*4)
        symmetry_score = symmetry_overlapse / area
        
        list_id.append(ID)
        list_data.append((symmetry_score,border_score,colour_cluster_score))
    
    df = pd.DataFrame(list_data,index=list_id,columns=["Asymmetry score","Border score","Cluster score"])

    return df

def print_accuracy(test_features,control_group,folds,classifiers):
    x_train, x_test, y_train, y_test = train_test_split(test_features, control_group,test_size=0.25, random_state=folds)
     
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    logreg = LogisticRegression()
    clf2 = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
    knn = KNeighborsClassifier()
    gnb = GaussianNB()
    lda = LinearDiscriminantAnalysis()
    svm = SVC()
    cent = NearestCentroid()
    
    logreg.fit(x_train, y_train)
    knn.fit(x_train, y_train)   
    gnb.fit(x_train, y_train)
    lda.fit(x_train, y_train)    
    svm.fit(x_train, y_train)
    cent.fit(x_train, y_train)
    
    def get_accuracy(x,y):
        a = logreg.score(x, y)
        b = clf2.score(x, y)
        c = knn.score(x, y)
        d = gnb.score(x, y)
        e = lda.score(x, y)
        f = svm.score(x, y)
        g = cent.score(x, y)
        
        return (a,b,c,d,e,f,g)

    training_sets = []
    test_sets = []
    
    for i in range(len(classifiers)):
        train = float(get_accuracy(x_train,y_train)[i])
        test = float(get_accuracy(x_test, y_test)[i])
        
        training_sets.append(train)
        test_sets.append(test)
       
    training_sets = tuple(training_sets)
    test_sets = tuple(test_sets)
    
    return (training_sets,test_sets)

def define_score():    
    value_data, control_groups = read_files()
    df = extract_info(value_data,control_groups)
    
    classifiers = ["Logistic regression","Decision Tree","Nearest Neighbor","Linear Discriminant Analysis",
                   "Gaussian Naive Bayes","Support Vector Machine","Nearest Centroid"]
    
    features = ["Asymmetry score","Border score","Cluster score"]
    test_features = df[features]
        
    iteration = []

    for folds in tqdm(range(len(df))):
        iteration.append(print_accuracy(test_features,control_groups,folds,classifiers))
        
    data = []
    
    for mode in range(2):  
        for types in range(len(classifiers)):
            value = 0
            for dataset in iteration:
                value += dataset[mode][types]
                
            value = value / len(iteration)
            
            data.append(value)
    
    mean_train = []
    mean_test = []
    
    for train in data[:7]:
        train = "{:0.2%}".format(train)
        mean_train.append(train)
        
    for test in data[7:]:
        test = "{:0.2%}".format(test)
        mean_test.append(test)
    
    mean_table = pd.DataFrame({"Types of classification:":classifiers,
                        "Mean training:":mean_train,
                        "Mean test:":mean_test})
    
    mean_table = mean_table.to_csv("classifiers.csv",index=False,sep=",")

define_score()


"""BOXPLOTS"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import math
import pandas as pd


def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_groups = []
    value_data = []
       
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines[:-1])
        control_groups.append(lines[-1])
    
    return value_data, control_groups

value_data, control_group = read_files()

list_colour_scores = []
list_border_score = []
list_symmetry_score = []
    
for tupl in value_data:
        
    border = float(tupl[2]) 
    area = float(tupl[1])
    symmetry_vertical = float(tupl[4])
    symmetry_horizontal = float(tupl[3])
    colour_score = tupl[5]
    colour_score = colour_score.replace(",", ".")
    colour_score = ((float(colour_score))) 
    
    border_score = (border**2) / (area*math.pi*4) 
    symmetry_score = (symmetry_vertical + symmetry_horizontal) / area     
    
    list_border_score.append(border_score)
    list_symmetry_score.append(symmetry_score)
    list_colour_scores.append(colour_score)
    
df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                    "Border score":list_border_score,
                    "Colour score":list_colour_scores})

X = X_full = df.to_numpy()
control_group = np.array(control_group)

X = PowerTransformer(method='box-cox').fit_transform(X)
X[:,0] = X[:,0]*2.0
X[:,1] = X[:,1]*2.0
X[:,2] = X[:,2]*1.6
symmetry = X[:, 0]
border = X[:, 1]
colour = X[:, 2]

true_list_border_score = []
true_list_symmetry_score = []
true_list_colour_cluster = []

false_list_border_score = []
false_list_symmetry_score = []
false_list_colour_cluster = []

for j in range(len(control_group)):
    if control_group[j] == "True":
        true_list_border_score.append(border[j])
        true_list_symmetry_score.append(symmetry[j])
        true_list_colour_cluster.append(colour[j])
            
    if control_group[j] == "False":
        false_list_border_score.append(border[j])
        false_list_symmetry_score.append(symmetry[j])
        false_list_colour_cluster.append(colour[j])

def print_plots(true,false, name):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot([true,false])
        ax.set_xticklabels( ['melanoma','non-melanoma'] )
        plt.title(name)
        plt.savefig(name)

for lists in [(true_list_border_score,false_list_border_score,"Border Score"),(true_list_symmetry_score,false_list_symmetry_score,"Asymmetry Score"),(true_list_colour_cluster,false_list_colour_cluster,"Color Score")]:
    print_plots(lists[0],lists[1],lists[2])
