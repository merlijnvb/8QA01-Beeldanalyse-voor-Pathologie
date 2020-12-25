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
    csv = open("labels.csv")
    csv_read = csv.readlines()
    csv.close()
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_groups = []
    value_data = []
    
    for lines in csv_read[1:]:
        lines = tuple(lines.rstrip().split(",")[:-1])
        control_groups.append(lines)
    
    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines)
    
    control_groups.sort()
    value_data.sort()
    
    return value_data, control_groups

def extract_info(value_data, control_groups):
    
    list_id = []
    list_data = []
    
    for tupl in value_data:
        ID = tupl[0]
        border = int(tupl[1])
        area = int(tupl[2])
        symmetry_vertical = int(tupl[3])
        symmetry_horizontal = int(tupl[4])
        colour_score = int(tupl[5])
        cluster_colour_score = float(tupl[6])
        
        diameter = border / math.pi
        
        border_score = (border**2) / (area*math.pi*4)
        symmetry_score = (symmetry_vertical + symmetry_horizontal) / area
        diameter_score = diameter / area
        
        list_id.append(ID)
        list_data.append((symmetry_score,border_score,colour_score,cluster_colour_score,diameter_score))
    
    df = pd.DataFrame(list_data,index=list_id,columns=["Asymmetry score","Border score","Colour score","Cluster score","Diameter score"])
    
    indexes = []
    control_list = []
    
    for data in control_groups:
        indexes.append(data[0])
        control_list.append(data[1])
           
    for id in df.index:
        if id not in indexes:
            df = df.drop(index=id)

    return df, control_list

def print_accuracy(test_features,control_group,folds,classifiers):
    x_train, x_test, y_train, y_test = train_test_split(test_features, control_group,test_size=100/len(control_group), random_state=folds)
     
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

def define_score(features):    
    value_data, control_groups = read_files()
    df, control_list = extract_info(value_data,control_groups)
    
    classifiers = ["Logistic regression","Decision Tree","Nearest Neighbor"," Linear Discriminant Analysis",
                   "Gaussian Naive Bayes","Support Vector Machine","Nearest Centroid"]
    
    features = features[1]
    name = features[0]
    test_features = df[features]
        
    iteration = []

    for folds in tqdm(range(len(df))):
        iteration.append(print_accuracy(test_features,control_list,folds,classifiers))
   
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
    
    mean_table = mean_table.to_csv("classifiers_{}.csv".format(name),index=False,sep=",")

def get_results():
    features = [("intervals",["Asymmetry score","Border score","Colour score","Diameter score"]),
                ("clusters",["Asymmetry score","Border score","Cluster score","Diameter score"])]
    
    for feature in features:
        define_score(feature)
