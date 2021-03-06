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

"""
    READ THE RESULTS FILE AND SAVE VALUES IN LISTS.
"""
def read_files():
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_groups = []
    value_data = []
    
    #PICK THE DATA FROM THE RESULTS FILE.
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines[:-1])
        control_groups.append(lines[-1])
    
    return value_data, control_groups

"""
    ORGANISE THE VALUES FURTHER AND MAKE SCORES.
"""
def extract_info(value_data, control_groups):
    
    list_id = []
    list_data = []
    
    #FORMAT THE DATA AND MAKE SCORES.
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

"""
    GET ACCURACY PER CLASSIFIER WHERE 40% OF THE DATA IS USED AS TRAINING DATA AND 60% OF THE DATA IS USED FOR TESTING THE TRAINED CLASSIFIER.
"""
def print_accuracy(test_features,control_group,folds,classifiers):
    #SPLITS SCORESETS IN TRAINING AND DATA VARIABLES.
    x_train, x_test, y_train, y_test = train_test_split(test_features, control_group,test_size=0.40, random_state=folds)
     
    #FIT SCORES
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #NAME THE CLASSIFIERS.
    logreg = LogisticRegression()
    clf2 = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
    knn = KNeighborsClassifier()
    gnb = GaussianNB()
    lda = LinearDiscriminantAnalysis()
    svm = SVC()
    cent = NearestCentroid()
    
    #FIT SCORES FOR CLASSIFIERS.
    logreg.fit(x_train, y_train)
    knn.fit(x_train, y_train)   
    gnb.fit(x_train, y_train)
    lda.fit(x_train, y_train)    
    svm.fit(x_train, y_train)
    cent.fit(x_train, y_train)
    
    """GET ACCURACY SCORE"""
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

"""
    DEFINE THE CLASSIFIER SCORES IN A MEAM TABLE.
"""
def define_score():    
    value_data, control_groups = read_files()
    df = extract_info(value_data,control_groups)
    
    classifiers = ["Logistic regression","Decision Tree","Nearest Neighbor","Linear Discriminant Analysis",
                   "Gaussian Naive Bayes","Support Vector Machine","Nearest Centroid"]
    
    features = ["Asymmetry score","Border score","Cluster score"]
    test_features = df[features]
    
    print(test_features)
        
    iteration = []
    
    #MAKE AS MANY FOLDS AS THERE IS DATASETS. 
    for folds in tqdm(range(len(df))):
        iteration.append(print_accuracy(test_features,control_groups,folds,classifiers))
        
    data = []
    
    #GET THE MEAN TRAINING AND TESTING RESULTS PER CLASSIEFIER.
    for mode in range(2):  
        for types in range(len(classifiers)):
            value = 0
            for dataset in iteration:
                value += dataset[mode][types]
                
            value = value / len(iteration)
            
            data.append(value)
    
    mean_train = []
    mean_test = []
    
    #UPDATE THE MEAN RESULTS TO PERCENTAGES.
    for train in data[:7]:
        train = "{:0.2%}".format(train)
        mean_train.append(train)
        
    for test in data[7:]:
        test = "{:0.2%}".format(test)
        mean_test.append(test)
    
    #POST THE PERCENTAGES FROM THE MEAN RESULTS IN A TABLE.
    mean_table = pd.DataFrame({"Types of classification:":classifiers,
                        "Mean training:":mean_train,
                        "Mean test:":mean_test})
    
    #SAVE THE TABLE
    mean_table = mean_table.to_csv("classifiers.csv",index=False,sep=",")

define_score()
