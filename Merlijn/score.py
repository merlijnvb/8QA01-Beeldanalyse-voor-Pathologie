import math
import pandas as pd
from tqdm import tqdm

def read_files():
    csv = open("labels.csv")
    csv_read = csv.readlines()
    csv.close()
    results = open("results.csv")
    results_read = results.readlines()
    results.close()
    
    control_group = []
    value_data = []
    
    for lines in csv_read[1:]:
        lines = tuple(lines.rstrip().split(","))
        control_group.append(lines[1])

    for lines in results_read[1:]:
        lines = lines.rstrip()
        lines = tuple(lines.split(","))
        value_data.append(lines)
        
    return value_data, control_group

def extract_info(value_data):
    
    list_colour_scores = []
    list_border_score = []
    list_symmetry_score = []
    list_diameter_score = []
    
    for tupl in value_data:
        border = int(tupl[1])
        area = int(tupl[2])
        symmetry_vertical = int(tupl[3])
        symmetry_horizontal = int(tupl[4])
        colour_score = float(tupl[-1])
        
        diameter = border / math.pi
        
        border_score = (border**2) / (area*math.pi*4)
        symmetry_score = (symmetry_vertical + symmetry_horizontal) / area
        diameter_score = diameter / area
        
        list_border_score.append(border_score)
        list_symmetry_score.append(symmetry_score)
        list_diameter_score.append(diameter_score)
        list_colour_scores.append(colour_score)
        
    df = pd.DataFrame({"Asymmetry score":list_symmetry_score,
                        "Border score":list_border_score,
                        "Colour score":list_colour_scores,
                        "Diameter score":list_diameter_score})   
    return df

def print_accuracy(test_features,control_group,folds,classifiers):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import NearestCentroid
    
    x_train, x_test, y_train, y_test = train_test_split(test_features, control_group, random_state=folds)
    
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
    
    for i in range(len(classifiers)):
        train = float(get_accuracy(x_train,y_train)[i])
        test = float(get_accuracy(x_test, y_test)[i])
        
        training_sets.append(train)
        test_sets.append(test)
       
    training_sets = tuple(training_sets)
    test_sets = tuple(test_sets)
    
    return (training_sets,test_sets)
    
def define_score():    
    value_data, control_group = read_files()
    df = extract_info(value_data)
    
    classifiers = ["Logistic regression","Decision Tree","Nearest Neighbor"," Linear Discriminant Analysis",
                   "Gaussian Naive Bayes","Support Vector Machine","Nearest Centroid"]
    
    test_features = df[list(df.keys())]
    
    iteration = []
    
    for folds in tqdm(range(len(value_data))):
        iteration.append(print_accuracy(test_features,control_group,folds,classifiers))
    
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
