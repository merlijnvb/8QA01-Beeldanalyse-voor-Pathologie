#https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html
#https://answers.opencv.org/question/204175/how-to-get-boundry-and-center-information-of-a-mask/
#https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
#https://www.marsja.se/pandas-scatter-matrix-pair-plot/#:~:text=A%20scatter%20matrix%20(pairs%20plot,method%20to%20visualize%20the%20dataset.

import math
import pandas as pd
from tqdm import tqdm

file = open("results.txt")
file_read = file.readlines()
file.close()

value_data = []

for lines in file_read[1:]:
    lines = lines.rstrip()
    lines = tuple(lines.split(","))
    value_data.append(lines)

list_borders = []
list_areas = []
list_symmetry_vertical = []
list_symmetry_horizontal = []
list_colour_scores = []
list_diameters = []
list_border_score = []
list_symmetry_score = []
list_diameter_score = []

lib = {}

for tupl in value_data:
    # print(tupl)
    border = int(tupl[1])
    area = int(tupl[2])
    symmetry_vertical = int(tupl[3])
    symmetry_horizontal = int(tupl[4])
    colour_score = float(tupl[-1])
    # print(colour_score)
    
    diameter = border / math.pi
    
    border_score = (border**2) / (area*math.pi*4)
    symmetry_score = symmetry_vertical / symmetry_horizontal
    diameter_score = diameter / area

    list_borders.append(border)
    list_areas.append(area)
    list_symmetry_vertical.append(symmetry_vertical)
    list_symmetry_horizontal.append(symmetry_horizontal)
    list_diameters.append(diameter)
    
    list_border_score.append(border_score)
    list_symmetry_score.append(symmetry_score)
    list_diameter_score.append(diameter_score)
    list_colour_scores.append(colour_score)
    
df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                   "Border score":list_border_score,
                   "Colour score":list_colour_scores,
                   "Diameter score":list_diameter_score})

pd.plotting.scatter_matrix(df, hist_kwds={'bins':len(value_data)},diagonal='kde')

# print(df.describe())

""" 
                SCORE EVALUATION
"""

def print_accuracy(x,y,folds,types):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import NearestCentroid
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=folds)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    
    clf2 = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)   
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)    
    svm = SVC()
    svm.fit(x_train, y_train)
    cent = NearestCentroid()
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
    
    for i in range(len(types)):
        train = float(get_accuracy(x_train,y_train)[i])
        test = float(get_accuracy(x_test, y_test)[i])
        
        training_sets.append(train)
        test_sets.append(test)
       
    training_sets = tuple(training_sets)
    test_sets = tuple(test_sets)
    
    return (training_sets,test_sets)
    
types = ["Logistic regression","Decision Tree","Nearest Neighbor"," Linear Discriminant Analysis",
         "Gaussian Naive Bayes","Support Vector Machine","Nearest Centroid"]

feature_names_x = ['Asymmetry score', 'Border score', 'Diameter score'] # add colour score here
x = df[feature_names_x]
y = df['Colour score'] # colour score has to be a banary number => 0 = non melenoma ; 1 = melanoma ; 2 = keratosis

iteration = []

for i in tqdm(range(len(value_data))):
    iteration.append(print_accuracy(x,y,i,types))

lgs_train = lgs_test = 0
dt_train = dt_test = 0
nn_train = nn_test = 0
lda_train = lda_test = 0
gnb_train = gnb_test = 0
svm_train = svm_test = 0
nc_train = nc_test = 0

for j in iteration:
    lgs_train += j[0][0]
    lgs_test += j[1][0]
    dt_train += j[0][1]
    dt_test += j[1][1]
    nn_train += j[0][2]
    nn_test += j[1][2]
    lda_train += j[0][3]
    lda_test += j[1][3]
    gnb_train += j[0][4]
    gnb_test += j[1][4]
    svm_train += j[0][5]
    svm_test += j[1][5]
    nc_train += j[0][6]
    nc_test += j[1][6]
    
mean_train_lgs = "{:0.2%}".format(lgs_train/len(iteration))
mean_test_lgs = "{:0.2%}".format(lgs_test/len(iteration))
mean_train_dt = "{:0.2%}".format(dt_train/len(iteration))
mean_test_dt = "{:0.2%}".format(dt_test/len(iteration))
mean_train_nn = "{:0.2%}".format(nn_train/len(iteration))
mean_test_nn = "{:0.2%}".format(nn_test/len(iteration))
mean_train_lda = "{:0.2%}".format(lda_train/len(iteration))
mean_test_lda = "{:0.2%}".format(lda_test/len(iteration))
mean_train_gnb = "{:0.2%}".format(gnb_train/len(iteration))
mean_test_gnb = "{:0.2%}".format(gnb_test/len(iteration))
mean_train_svm = "{:0.2%}".format(svm_train/len(iteration))
mean_test_svm = "{:0.2%}".format(gnb_test/len(iteration))
mean_train_nc = "{:0.2%}".format(nc_train/len(iteration))
mean_test_nc = "{:0.2%}".format(nc_test/len(iteration))

training = [mean_train_lgs,mean_train_dt,mean_train_nn,mean_train_lda,mean_train_gnb,mean_train_svm,mean_train_nc]
test = [mean_test_lgs,mean_test_dt,mean_test_nn,mean_test_lda,mean_test_gnb,mean_test_svm,mean_test_nc]
    
mean_table = pd.DataFrame({"Types of classification:":types,
                   "Mean training:":training,
                   "Mean test:":test})

print(mean_table)