import sys
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import math

def ordinal_encoder(categories, a):
    cats = len(categories)*[None]
    for i in range(len(categories)):
        cat = {}
        c = categories[i]
        for j in range(len(c)):
            cat[c[j]] = j
        cats[i] = cat

    w, h = a.shape
    t = np.empty(a.shape, dtype=float)
    for i in range(w):
        for j in range(h):
            t[i][j] = cats[j][a[i][j]]
    return t

def main(argv):
    argc = len(argv)
    # dt - Decision trees
    # nn - Neural nets
    # svm - Support vector machines
    # log - Logistic regression
    models = ["dt", "nn", "svm", "log"]
    # default it is a decision tree
    model = "dt"
    X1 = None
    X2 = None
    y1 = None
    y2 = None
    X1_t = None
    X2_t = None
    out_file = None
    
    for i in range(1, argc):
        if argv[i] == "-train" and (i+1) < argc:
            try:
                d_train = np.genfromtxt(argv[i+1], dtype=str, delimiter=',', skip_header=1, usecols=range(2, 184))
                d_train[d_train == ''] = '0'
                y1 = (d_train[:,0]).astype(np.uint8)
                y2 = (d_train[:,1]).astype(float)
                # Select subset of features for classification
                # The features chosen were based on my knowledge after looking at the data and making some reasonable assumptions 
                col_subset1 = (list(range(2, 6)) + [8, 9]
                            + list(range(11, 18)) + [28]
                            + list(range(32, 36)) + list(range(37, 52))
                            + list(range(55, 93)) + list(range(95, 107))
                            + list(range(114, 181)))
                X1 = d_train[:, col_subset1]

                col_subset2 = [8, 10] + list(range(18, 24)) + [26, 27, 53, 54, 93, 94] + list(range(107, 114))
                X2 = d_train[:, col_subset2]
            except:
                print("Invalid training file.")
                return

        if argv[i] == "-test" and (i+1) < argc:
            try:
                d_test = np.genfromtxt(argv[i+1], dtype=str, delimiter=',', skip_header=1, usecols=range(1, 184))
                d_test[d_test == ''] = '0'
                # Select subset of features for classification
                # The features chosen were based on my knowledge after looking at the data and making some reasonable assumptions 
                col_subset1_t = (list(range(1, 5)) + [7, 8]
                               + list(range(10, 17)) + [27]
                               + list(range(31, 35)) + list(range(36, 51))
                               + list(range(54, 92)) + list(range(94, 106))
                               + list(range(113, 180)))
                X1_t = d_test[:, col_subset1_t]

                col_subset2_t = [7, 9] + list(range(17, 23)) + [25, 26, 52, 53, 92, 93] + list(range(106, 113))
                X2_t = d_test[:, col_subset2_t]
            except:
                print("Invalid testing file.")


        elif argv[i] == "-m" and (i+1) < argc:
            try:
                model = argv[i+1]
                if model not in models:
                    raise ValueError
            except:
                print("Invalid model selected.")
                return

        elif argv[i] == "-o" and (i+1) < argc:
            out_file = argv[i+1]

    if (X1 is None) or (X1_t is None) or (out_file is None):
        print("Please specify training and testing data together with output file using: -train <file_path> -test <file_path> -o <file_path>")
        return
    
    # decision tree
    if model == "dt":
        # encode categorical data
        # perform classification
        X1 = np.concatenate((ordinal_encoder([["Unmanaged", "Onboarding", "Retention"], ["Unconfirmed", "In Progress", "Active", "Enrolled"],
                                             ["Business", "Trade"], ["CA", "US"], ["None", "1", "2to5", "6to10", "11to50", "50plus"],
                                             ["None", "1to2", "3to5", "6to10", "11to25", "25plus"], ["other", "directOther", "directEIN", "email", "phone", "liveTransfer"]],
                                             X1[:, 0:7]), (X1[:, 7:]).astype(float)), axis=1)
        dt_class = tree.DecisionTreeClassifier()
        clf = GridSearchCV(dt_class, {"max_depth": list(range(3, 101))}, scoring="roc_auc", iid=False, refit=True, cv=5, error_score=np.nan)
        clf.fit(X1, y1)
        print(f"Depth of tree that achieved max classification performance: {(clf.best_params_)['max_depth']}")
        print(f"ROC-AUC score of best performing tree on training data: {clf.best_score_}")
        # predict the labels using the best performing tree
        p_labels = clf.predict(X1)
        correct = 0
        for i, j in zip(y1, p_labels):
            if i == j:
                correct += 1
        print(f"Accuracy of the labels for the best classifying tree on training data: {(100*correct/len(y1)):.3g}%")

        # perform regression
        X2 = np.concatenate((ordinal_encoder([["None", "1", "2to5", "6to10", "11to50", "50plus"],
                                             ["None", "lessthan1", "1to5", "5to25", "25to100", "100plus"]],
                                             X2[:, 0:2]), (X2[:, 2:]).astype(float)), axis=1)
        dt_regress = tree.DecisionTreeRegressor()
        regressor = GridSearchCV(dt_regress, {"max_depth": list(range(3, 101))}, scoring="neg_mean_squared_error", iid=False, refit=True, cv=5, error_score=np.nan)
        regressor.fit(X2, y2)
        print(f"Depth of tree that achieved max regression performance: {(regressor.best_params_)['max_depth']}")
        print(f"Root-mean-square error (RMSE) score of best performing tree on training data: {math.sqrt(abs(regressor.best_score_))}")
        # regression on the partial data for the ones that were classified as '1' in the previous step
        p_regress = regressor.predict(X2[p_labels > 0])
        sum_square_diff = 0
        for i, j in zip(y2, p_regress):
            sum_square_diff += (i-j)**2
        print(f"RMSE of the tree on training data: {(math.sqrt(sum_square_diff/len(y2))):.3g}")            

        p_labels_t = clf.predict(X1_t)                
        p_regress = regressor.predict(X2_t[p_labels_t > 0])
        with open(out_file, 'w+') as f_out:
            
    
    #elif model == "nn":
        
    
if __name__ == '__main__':
    main(sys.argv)
