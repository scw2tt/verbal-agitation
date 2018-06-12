"""
Sean Wolfe 
6/9/18
Filename: classifyAudio.py


"""

import sklearn as skl
from sklearn import neighbors as k
from sklearn import svm
from sklearn import cross_validation as cross_v
import numpy as np
import pandas as pd
import scipy as sci
import os
import random as rd
from sklearn.feature_selection import RFE

# file locations
LOC_HEADER = "labData\headerF.txt"
DIR_AGI = "labData\Agitation"
DIR_NON_AGI = "labData\NonAgitation"


# main program
def detectAgi(agiFolder, nonAgiFolder, silenceRemoval = False, classifier = "knn"):
    """
    This function trains and tests a classifier on the audio data that consists of 
    34 features.

    ARGUMENTS:
        agiFolder - folder name of the txt files that contain the agitation-labelled data
        nonAgiFolder - folder for the non-agitation labelled data
        silenceRemoval - if true, the silence removal algorithm will filter the dataset
        classifier - the classifier that will be used, knn by default

    OUTPUT:
        x_columns - the pandas data frame of the featured audio data
        y_column  - the pandas data frame for the classification of these
    """

    #----------------------------------------------------------------------------
    # segment the text data into the sets: agitation and not agitation
    head = pd.read_table(LOC_HEADER, sep= ',', header = 2)
    agi = pd.DataFrame(columns = head.columns)
    nonAgi = pd.DataFrame(columns = head.columns)
    for filename in os.listdir(agiFolder):
        features = pd.read_table(DIR_AGI +  "\\" + filename, sep= ',', header=2)
        agi = agi.append(features)

    for filename in os.listdir(nonAgiFolder):
        features = pd.read_table(DIR_NON_AGI +  "\\" + filename, sep= ',', header=2)
        nonAgi = nonAgi.append(features)

    # classify 
    agi['Agitation'] = 'True'
    nonAgi['Agitation'] = 'False'
    df = agi.append(nonAgi)
    # The columns that we will be making predictions with.
    x_columns = df[df.columns[1: len(df.columns) - 2]]
    # The column that we want to predict.
    y_column = df["Agitation"]


    # split the dataset into training and testing
    X_trn, X_tst, y_trn, y_tst = cross_v.train_test_split(x_columns, y_column, test_size=0.2, random_state=65)
    print("---------train-----------")
    print(X_trn.shape)
    print("---------test------------")
    print(X_tst.shape)

    #----------------------------------------------------------------------------
    # train the model (supervised learning)
    if classifier == "knn":
        model = k.KNeighborsClassifier(n_neighbors = 5)
    else:
        model = svm.LinearSVC()
    
    model.fit(X_trn, np.ravel(y_trn))

    #----------------------------------------------------------------------------
    # test the model
    print("Score: " + str(model.score(X_tst, y_tst)))
    return x_columns, y_column

X, y = detectAgi(DIR_AGI, DIR_NON_AGI, classifier= "svc")


# principle component analysis
def audioPCA():

    # loop through 1 through 34 features
    
        # perform train and test with those features



    return


# recursive feature elimination
def audioRFE(X, y):
    # run the RFE to obtain the rankings of the features
    model = svm.LinearSVC()
    rfe = RFE(model, 1)
    fit = rfe.fit(X,y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_

    return fit.ranking_

ranking_list = audioRFE(X,y)

# take the features list and correlate to the names
X_cols = X.columns
print("--features--")
print(X_cols)
feat_ranking = []
for j in range(34):
    feat_ranking.append("-1")

for i in range(len(feat_ranking)):
    feature = X_cols[i]
    ranking = feat_ranking[i]
    feat_ranking[i - 1] = feature



for i in range(1,len(feat_ranking)):
    x_columns = X[feat_ranking[0:i]]
    # split the dataset into training and testing
    X_trn, X_tst, y_trn, y_tst = cross_v.train_test_split(x_columns, y, test_size=0.2, random_state=65)
    #----------------------------------------------------------------------------
    # train the model (supervised learning)
    model = svm.LinearSVC()
    model.fit(X_trn, np.ravel(y_trn))

    #----------------------------------------------------------------------------
    # test the model
    
    print("-----------")
    print("-----------")
    print("  ")
    print(str(i+1) + " features score: " + str(model.score(X_tst, y_tst)))
    print("features used:")
    for feature in feat_ranking[0:i]:
        print(feature)

    print("  ")



