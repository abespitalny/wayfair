import sys
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import math
import csv

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
    X1_tr = None
    X2_tr = None
    y1_tr = None
    y2_tr = None
    cuids_te = None
    X1_te = None
    X2_te = None
    # This is the file used to write predictions to:
    out_file = None

    # Select subsets of features for classification and regression.
    # The features chosen were based on my knowledge after looking at the data and making some reasonable assumptions.
    # Columns used for classification:
    col_subset_class = np.concatenate((np.array([2, 4, 5]), np.arange(7, 10), np.arange(11, 18),
                                       np.array([28]), np.arange(32, 36), np.arange(37, 52),
                                       np.arange(55, 93), np.arange(95, 107), np.arange(114, 181)))
    # Columns used for regression:
    col_subset_regress = np.concatenate((np.array([8, 10]), np.arange(18, 24), np.array([26, 27, 53, 54, 93, 94]), np.arange(107, 114)))

    for i in range(1, argc):
        if argv[i] == "-train" and (i+1) < argc:
            try:
                d_train = np.genfromtxt(argv[i+1], dtype=str, delimiter=',', skip_header=1, usecols=range(2, 184))
                # Fill missing values with 0.
                d_train[d_train == ''] = '0'
                y1_tr = (d_train[:,0]).astype(np.uint8)
                y2_tr = (d_train[:,1]).astype(float)
                X1_tr = d_train[:,col_subset_class]
                X2_tr = d_train[:,col_subset_regress]
            except:
                print("Invalid training file.")
                return

        if argv[i] == "-test" and (i+1) < argc:
            try:
                d_test = np.genfromtxt(argv[i+1], dtype=str, delimiter=',', skip_header=1, usecols=range(1, 182))
                d_test[d_test == ''] = '0'
                cuids_te = d_test[:,0]
                X1_te = d_test[:,col_subset_class-1]
                X2_te = d_test[:,col_subset_regress-1]
            except:
                print("Invalid testing file.")
                return

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

    if (X1_tr is None) or (X1_te is None) or (out_file is None):
        print("Please specify training and testing data together with output file using: -train <file_path> -test <file_path> -o <file_path>")
        return

    # Decision tree:
    if model == "dt":
        #### Train classification model ####
        # I needed to manually specify the order of the categories since sci-kit was just using sorted order,
        # but some of the features have an inherent order such as number of purchases per year which is important for decision trees
        # Category names used for classification
        cats_class = [["Unmanaged", "Onboarding", "Retention"],
                      ["Business", "Trade"], ["CA", "US"],
                      ["None", "Other", "Primary", "Purchaser"],
                      ["None", "1", "2to5", "6to10", "11to50", "50plus"],
                      ["None", "1to2", "3to5", "6to10", "11to25", "25plus"],
                      ["other", "directOther", "directEIN", "email", "phone", "liveTransfer"]]
        # Category names used for regression
        cats_regress = [["None", "1", "2to5", "6to10", "11to50", "50plus"], ["None", "lessthan1", "1to5", "5to25", "25to100", "100plus"]]

        # Encode categorical data:
        X1_tr = np.concatenate((ordinal_encoder(cats_class, X1_tr[:,0:7]), (X1_tr[:,7:]).astype(float)), axis=1)
        # Using information gain rather than gini index caused the classifier to perform worse
        dt_class = tree.DecisionTreeClassifier()
        clf = GridSearchCV(dt_class, {"max_depth":np.arange(3, 101)}, scoring="roc_auc", iid=False, refit=True, cv=5, error_score=np.nan)
        clf.fit(X1_tr, y1_tr)
        print(f"Depth of tree that achieved max classification performance on training data: {(clf.best_params_)['max_depth']}")
        print(f"ROC-AUC score of best performing tree on training data: {clf.best_score_:.4g}")

        # Predict the labels using the best performing tree
        p_labels_tr = clf.predict(X1_tr)
        correct = 0
        for i, j in zip(y1_tr, p_labels_tr):
            if i == j:
                correct += 1
        print(f"Accuracy of the predicted labels for the best classifying tree on training data: {(100*correct/len(y1_tr)):.4g}%")

        #### Train regression model ####
        # We train the regression model on all the data, but predict on only the ones that were classified as '1' in the classification step
        X2_tr = np.concatenate((ordinal_encoder(cats_regress, X2_tr[:,0:2]), (X2_tr[:,2:]).astype(float)), axis=1)
        dt_regress = tree.DecisionTreeRegressor()
        regressor = GridSearchCV(dt_regress, {"max_depth":np.arange(3, 101)}, scoring="neg_mean_squared_error", iid=False, refit=True, cv=5, error_score=np.nan)
        regressor.fit(X2_tr, y2_tr)
        print(f"Depth of tree that achieved max regression performance on training data: {(regressor.best_params_)['max_depth']}")
        # Note: regression score calculated by sci-kit is the negation of the mean-squared error
        print(f"Root-mean-square error (RMSE) score of best performing tree on training data: {math.sqrt(abs(regressor.best_score_)):.4g}")

        # Regression performance on the data classified as '1' in the classification step
        p_regress_tr = regressor.predict(X2_tr[p_labels_tr == 1])
        sum_square_diff = 0
        k = 0
        for i, j in zip(p_labels_tr, y2_tr):
            if i == 1:
                sum_square_diff += (p_regress_tr[k] - j)**2
                k += 1
            else:
                # Our prediction for the regression is 0 since the classification is '0'
                sum_square_diff += j**2
        print(f"RMSE of the tree on training data: {(math.sqrt(sum_square_diff/len(y2_tr))):.4g}")

        X1_te = np.concatenate((ordinal_encoder(cats_class, X1_te[:,0:7]), (X1_te[:,7:]).astype(float)), axis=1)
        p_labels_te = clf.predict(X1_te)
        X2_te = np.concatenate((ordinal_encoder(cats_regress, X2_te[:,0:2]), (X2_te[:,2:]).astype(float)), axis=1)
        p_regress_te = regressor.predict(X2_te[p_labels_te > 0])

        # Write predictions to the specified output file
        with open(out_file, 'w+') as f_out:
            preds = np.concatenate((np.vstack(p_labels_te), np.vstack(np.empty(len(p_labels_te)))), axis=1)
            j = 0
            for i in preds:
                if i[0] == 1:
                    i[1] = round(p_regress_te[j], 2)
                    j += 1
                else:
                    i[1] = 0

            out_data = np.concatenate((np.vstack(cuids_te), preds), axis=1)
            writer = csv.writer(f_out)
            writer.writerow(["cuid", "pred_convert_30", "pred_revenue_30"])
            writer.writerows(out_data)

    elif model == "nn":
        #### Train classification model ####
        # Encode categorical data using one-hot encoding:
        X1_tr = np.concatenate((OneHotEncoder(sparse=False).fit_transform(X1_tr[:,0:7]), (X1_tr[:,7:]).astype(float)), axis=1)
        # Scale data so it has 0 mean and 1 variance. Use this same scaling on the test data as well
        scaler_class = StandardScaler(copy=False)
        scaler_class.fit(X1_tr)
        scaler_class.transform(X1_tr)
        
        
    
if __name__ == '__main__':
    main(sys.argv)
