import sys
import math
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

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
    models = ["dt", "nn", "svm"]
    names = {"dt":"tree", "nn":"neural net", "svm":"support vector machine"}
    # default it is a decision tree
    model_class = "dt"
    model_regress = "dt"
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

        elif argv[i] == "-mc" and (i+1) < argc:
            try:
                model_class = argv[i+1]
                if model_class not in models:
                    raise ValueError
            except:
                print("Invalid classification model selected.")
                return

        elif argv[i] == "-mr" and (i+1) < argc:
            try:
                model_regress = argv[i+1]
                if model_regress not in models:
                    raise ValueError
            except:
                print("Invalid regression model selected.")
                return

        elif argv[i] == "-o" and (i+1) < argc:
            out_file = argv[i+1]

    if (X1_tr is None) or (X1_te is None) or (out_file is None):
        print("Please specify training and testing data together with output file using: -train <file_path> -test <file_path> -o <file_path>")
        return

    #### Train classification model ####
    clf = None
    if model_class == "dt":
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

        # Encode categorical data for both training and testing data:
        X1_tr = np.concatenate((ordinal_encoder(cats_class, X1_tr[:,0:7]), (X1_tr[:,7:]).astype(float)), axis=1)
        X1_te = np.concatenate((ordinal_encoder(cats_class, X1_te[:,0:7]), (X1_te[:,7:]).astype(float)), axis=1)
        # Using information gain rather than gini index caused the classifier to perform worse
        clf = GridSearchCV(DecisionTreeClassifier(criterion="gini"), {"max_depth":np.arange(3, 101)}, scoring="roc_auc", iid=False, refit=True, cv=5, error_score=np.nan)
        clf.fit(X1_tr, y1_tr)
        print(f"Depth of tree that achieved max classification performance on training data: {(clf.best_params_)['max_depth']}")

    elif model_class == "nn":
        # Encode categorical data using one-hot encoding:
        encoder = OneHotEncoder(sparse=False)
        X1_tr = np.concatenate((encoder.fit_transform(X1_tr[:,0:7]), (X1_tr[:,7:]).astype(float)), axis=1)
        X1_te = np.concatenate((encoder.transform(X1_te[:,0:7]), (X1_te[:,7:]).astype(float)), axis=1)
        # Scale data so it has 0 mean and 1 variance. Use this same scaling on the test data as well
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X1_tr)
        scaler.transform(X1_te)
        # I'm having the search use all the CPUs since training the neural net must be done in the cloud with sufficient computing power
        clf = GridSearchCV(MLPClassifier(activation="relu", solver="adam"), {"alpha":10.0**-np.arange(-1, 7), "hidden_layer_sizes":np.arange(5, 100)}, scoring="roc_auc",
                           iid=False, refit=True, cv=5, error_score=np.nan, n_jobs=-1)
        clf.fit(X1_tr, y1_tr)
        print(f"Hyperparamter alpha and number of nodes in hidden layer of neural net that achieved max classification performance on training data: {clf.best_params_}")

    else:
        return

    print(f"ROC-AUC score of best performing {names[model_class]} on training data: {clf.best_score_:.4g}")
    # Predict the labels using the best performing estimator
    p_labels_tr = clf.predict(X1_tr).astype(np.uint8)
    correct = 0
    for i, j in zip(y1_tr, p_labels_tr):
        if i == j:
            correct += 1
    print(f"Accuracy of the predicted labels for the best classifying {names[model_class]} on training data: {(100*correct/len(y1_tr)):.4g}%")

    #### Train regression model ####
    regressor = None
    # We train the regression model on all the data, but predict on only the ones that were classified as '1' in the classification step
    if model_regress == "dt":
        X2_tr = np.concatenate((ordinal_encoder(cats_regress, X2_tr[:,0:2]), (X2_tr[:,2:]).astype(float)), axis=1)
        X2_te = np.concatenate((ordinal_encoder(cats_regress, X2_te[:,0:2]), (X2_te[:,2:]).astype(float)), axis=1)
        regressor = GridSearchCV(DecisionTreeRegressor(), {"max_depth":np.arange(3, 101)}, scoring="neg_mean_squared_error", iid=False, refit=True, cv=5, error_score=np.nan)
        regressor.fit(X2_tr, y2_tr)
        print(f"Depth of tree that achieved max regression performance on training data: {(regressor.best_params_)['max_depth']}")

    elif model_regress == "nn":
        encoder = OneHotEncoder(sparse=False)
        X2_tr = np.concatenate((encoder.fit_transform(X2_tr[:,0:2]), (X2_tr[:,2:]).astype(float)), axis=1)
        X2_te = np.concatenate((encoder.transform(X2_te[:,0:2]), (X2_te[:,2:]).astype(float)), axis=1)
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X2_tr)
        scaler.transform(X2_te)
        regressor = GridSearchCV(MLPRegressor(activation="relu", solver="adam"), {"alpha":10.0**-np.arange(-1, 7), "hidden_layer_sizes":np.arange(5, 100)}, scoring="neg_mean_squared_error",
                                 iid=False, refit=True, cv=5, error_score=np.nan, n_jobs=-1)
        regressor.fit(X2_tr, y2_tr)
        print(f"Hyperparamter alpha and number of nodes in hidden layer of neural net that achieved max regression performance on training data: {regressor.best_params_}")

    else:
        return

    # Note: regression score calculated by sci-kit is the negation of the mean-squared error
    print(f"Root-mean-square error (RMSE) score of best performing {names[model_regress]} on training data: {math.sqrt(abs(regressor.best_score_)):.4g}")
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
    print(f"RMSE of the {names[model_regress]} on training data: {(math.sqrt(sum_square_diff/len(y2_tr))):.4g}")

    # Predict on the testing data:
    p_labels_te = clf.predict(X1_te).astype(np.uint8)
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


if __name__ == '__main__':
    main(sys.argv)
