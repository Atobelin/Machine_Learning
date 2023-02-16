from sklearn import model_selection
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np


RANDOM_STATE = 1202208


def loaddata():
    df = pd.read_csv("data/Churn_Modelling.csv")
    # Eliminate irrelevant columns(RowNumber, Surname, CustomerId)
    df = df.drop(columns=['RowNumber', 'Surname', 'CustomerId'])
    return df


def dealingWithMissingValues(df):
    # There is no missing values in this dataset
    print("=========================================")
    print(df.isnull().sum())
    print("There is no missing values in this dataset")
    print("=========================================\n")
    return df


def handlingCategoricalData(df):
    # Use one hot encoder to categotacal columns: Geography and Gender
    encoder = OneHotEncoder(sparse=False)
    categEncoded = encoder.fit_transform(df[["Geography", "Gender"]])
    feature_names = encoder.get_feature_names_out(["Geography", "Gender"])
    df1 = pd.DataFrame(data=categEncoded, columns=feature_names)
    df = df.drop(columns=['Geography', 'Gender'])
    result = pd.concat([df, df1], axis=1)
    return result


def splitData(df):
    # split dataset to features and target
    target = df['Exited']
    features = df.drop(columns=['Exited'])
    return (features, target)


def dealingWithOutliers(features, target):
    # use Isolation Forest to deal with outliers
    features = features.to_numpy()
    clf = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    clf.fit(features)
    results = clf.predict(features)
    normal_features = features[results == 1]
    normal_target = target[results == 1]
    print("=========================================")
    print("There are", target.size-normal_target.size, "Outliers in the data")
    print("=========================================\n")
    return (normal_features, normal_target)


def scalingData(df):
    # standardizing data
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

# useImbalance
# 1 -> SMOTE
# 2 -> BorderlineSMOTE


def handlingImbalance(features, target, useImbalance):
    # use SMOTE to handle imbalance
    print("=========================================")
    print("imbalance data")
    print(Counter(target))
    print("=========================================\n")
    if useImbalance == 2:
        sm = BorderlineSMOTE(random_state=RANDOM_STATE)
    else:
        sm = SMOTE(random_state=RANDOM_STATE)
    features, target = sm.fit_resample(features, target)
    print("=========================================")
    print("rebalance data")
    print(Counter(target))
    print("=========================================\n")
    return (features, target)

# useImbalance
# 0 -> no
# 1 -> SMOTE
# 2 -> BorderlineSMOTE
# default is 2


def preprocessing(data, useImbalance=2):
    # baseline preprocess
    data = dealingWithMissingValues(data)
    data = handlingCategoricalData(data)
    features, target = splitData(data)
    features, target = dealingWithOutliers(features, target)
    features = scalingData(features)
    if useImbalance != 0:
        features, target = handlingImbalance(features, target, useImbalance)
    return (features, target)


def runClassifiers(features, target):
    # compare 5 classifiers and find two bests.
    # five classifiers
    clfs = [KNeighborsClassifier(), DecisionTreeClassifier(),
            GaussianNB(), SVC(), RandomForestClassifier()]
    # best scores and best clfs
    bests = [0, 0]
    best_clfs = clfs[:2]
    print("=========================================")
    for clf in clfs:
        # use cross fold validation here
        scores = model_selection.cross_val_score(clf, features, target, cv=5)
        mean = scores.mean()
        print(clf, mean)
        if mean > min(bests):
            index = 0 if bests[0] < bests[1] else 1
            bests[index] = mean
            best_clfs[index] = clf
    print(best_clfs)
    print("=========================================\n")
    return best_clfs


def hyperparameter_optimization(features, target, best_clfs):
    # Based on the above output, it is clear
    # best_clfs is [RandomForestClassifier(), KNeighborsClassifier()]
    # RandomForestClassifier
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=100)]
    max_features = ['log2', 'sqrt']
    min_samples_split = [2, 4, 6]
    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'min_samples_split': min_samples_split
    }
    clf = model_selection.GridSearchCV(
        best_clfs[0], random_grid, cv=5, n_jobs=-1)
    clf.fit(features, target)
    print("=========================================")
    print(clf.best_params_, "with a score of ", clf.best_score_)
    # KNeighborsClassifier
    n_neighbors = list(range(1, 20))
    p = [1, 2, 3]
    # Create the knn grid
    knn_grid = {
        'n_neighbors': n_neighbors,
        'p': p
    }
    clf = model_selection.GridSearchCV(best_clfs[1], knn_grid, cv=5)
    clf.fit(features, target)
    print(clf.best_params_, "with a score of ", clf.best_score_)
    print("=========================================\n")


def runModel(features, target):
    # use hyperparametric optimized models here. the parameters are the result in baseline
    clfs = [KNeighborsClassifier(n_neighbors=1, p=2), RandomForestClassifier(
        max_features='log2', min_samples_split=2, n_estimators=621)]
    means = []
    # also use cross val score
    print("=========================================")
    for clf in clfs:
        scores = model_selection.cross_val_score(clf, features, target, cv=5)
        mean = scores.mean()
        means.append(mean)
    print("KNN:", means[0], ", RandomForest:", means[1])
    print("=========================================\n")
    return means


def feature_selection(features, target):
    # extra RandomForestClassifier
    # find the importances and sort
    forest = RandomForestClassifier(
        n_estimators=250, random_state=RANDOM_STATE)
    forest.fit(features, target)
    importances = forest.feature_importances_
    sortedIndices = np.argsort(importances)
    numberOfFeatures = []
    accurK = []
    accurR = []
    # delete 0,1,2,3 features and show the result in plot
    for num in range(0, 4):
        numberOfFeatures.append(num)
        indicesToDelete = sortedIndices[0:num+1]
        features_new = np.delete(features, indicesToDelete, axis=1)
        accur = runModel(features_new, target)
        accurK.append(accur[0])
        accurR.append(accur[1])

    plt.figure()
    plt.xlabel("Number of features removed")
    plt.ylabel("KRR Cross validation score")
    plt.plot(numberOfFeatures, accurK)
    plt.show()

    plt.figure()
    plt.xlabel("Number of features removed")
    plt.ylabel("RandomForest Cross validation score")
    plt.plot(numberOfFeatures, accurR)
    plt.show()


def baseline():
    data = loaddata()
    features, target = preprocessing(data)
    best_clfs = runClassifiers(features, target)
    hyperparameter_optimization(features, target, best_clfs)


def experimentation():
    data = loaddata()
    features, target = preprocessing(data)
    feature_selection(features, target)


def research():
    data = loaddata()
    features, target = preprocessing(data, useImbalance=0)
    print("use no rebalance")
    runModel(features, target)
    features, target = preprocessing(data, useImbalance=1)
    print("use SMOTE rebalance")
    runModel(features, target)
    features, target = preprocessing(data, useImbalance=2)
    print("use BorderlineSMOTE rebalance")
    runModel(features, target)


research()
