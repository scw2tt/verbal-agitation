"""
Sean Wolfe 
6/9/18
Filename: classifyAudio.py

Contains functions for wrangling and analyzing audio data in a txt format
"""

import sklearn as skl
from sklearn import neighbors as k
from sklearn import svm
from sklearn import cross_validation as cross_v
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
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


def detectAgi(agiFolder, nonAgiFolder, classifier = "svm"):
    """Coverts the audio features in the text files into a pandas dataframe. Then a
    classifier (either k-nearest neighbor or SVM) will train and test on that data. 
    The score of the algorithm will be printed out.
    
    Arguments:
        agiFolder {string} -- location of the folder that holds agitation-labelled data
        nonAgiFolder {string} -- location fo the folder that holds the other data
    
    Keyword Arguments:
        classifier {str} -- if "knn", then k-nearest neighbor will be used as the classifier, 
                            if not then the support vector machine will be used (default: {"knn"})
                            "rfc" : Random Forest Classifier
                            "knn" : k- Nearest Neighbor where k = 5
                            "tree" : Decision Tree Classifier
    
    Returns:
         X {pandas dataframe} -- the features columns from both folders combined into a pandas dataset
         y {pandas array} -- the labels that go along with the features for each row
    """


    #----------------------------------------------------------------------------
    # segment the text data into the sets: agitation and not agitation
    head = pd.read_table(LOC_HEADER, sep= ',', header = 2)
    agi = pd.DataFrame(columns = head.columns)
    nonAgi = pd.DataFrame(columns = head.columns)
    # loop through both directors and add the data to the data frame
    for filename in os.listdir(agiFolder):
        
        features = pd.read_table(DIR_AGI +  "\\" + filename, sep= ',', header=2)
        agi = agi.append(features)

    for filename in os.listdir(nonAgiFolder):
        features = pd.read_table(DIR_NON_AGI +  "\\" + filename, sep= ',', header=2)
        nonAgi = nonAgi.append(features)

    # add the class as a column to the datasets
    agi['Agitation'] = 'True'
    nonAgi['Agitation'] = 'False'
    df = agi.append(nonAgi)

    # The columns that we will be making predictions with.
    X = df[df.columns[1: len(df.columns) - 2]]
    # The column that we want to predict.
    y = df["Agitation"]


    # split the dataset randomly into training and testing
    seed = rd.randint(1,100)
    X_trn, X_tst, y_trn, y_tst = cross_v.train_test_split(X, y, test_size=0.2, random_state=seed)
    print("---------train size-----------")
    print(X_trn.shape)
    print("---------test size------------")
    print(X_tst.shape)

    #----------------------------------------------------------------------------
    # train the model (supervised learning)
    # use either k-nearest neighbors or support vector machine as the model
    if classifier == "tree":
        model = tree.DecisionTreeClassifier()
    elif classifier == "rfc":
        model = RFC() 
    elif classifier == "knn":
        model = k.KNeighborsClassifier(n_neighbors = 5)
    else:
        model = svm.LinearSVC()
    
    model.fit(X_trn, np.ravel(y_trn))

    #----------------------------------------------------------------------------
    # test the model
    print("Score: " + str(model.score(X_tst, y_tst)))
    return X, y



def audioRFE(X, y):
    """Performs the Recursive Feature Elimination algorithm from the sklearn python 
     module for machine learning on the dataset in order to rank the features in order 
     of importance for the purpose of dimensionality reduction. SVC is used in this case.
    
    Arguments:
        X {[float][float]} -- pandas data frame of the input features to the model
        y {[string]} -- pandas array of the classification of the rows of X
    
    Returns:
        ranking_list [int] -- rankings for each feature in the list. For example, an
                            input of (height, weight, wingspan) that outputs to 
                            [1,3,2] means that the order of importance of the features
                            is: height, wingspan, weight
    """


    # run the RFE to obtain the rankings of the features
    model = svm.LinearSVC()
    rfe = RFE(model, 1)
    fit = rfe.fit(X,y)
    # print out the report
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    print(" ")
    print(" ")
    return fit.ranking_

def RFE_Analysis(X, y, ranking_list):
    """Iteratively trains and tests the dataset using SVM. It starts by just using the 
    most important feature to train and test on the dataset. Then, it successively adds
    the next most important feature and tests its performance, printing out the features 
    used and the score each time.
    
    Arguments:
        X {[float][float]} -- input features in the pandas dataframe format
        y {[string]} -- labels for the features of X
        ranking_list {[int]} -- list of the rankings for each column in X
    """


    X_cols = X.columns
    feat_ranking = []
    # fill the initial ranking array with all ones
    for j in range(len(X.columns)):
        feat_ranking.append("-1")

    # fill the array with the features in their order of importance
    for i in range(len(feat_ranking)):
        feature = X_cols[i]
        ranking = ranking_list[i]
        feat_ranking[ranking - 1] = feature

    # print out the ranking of the features
    print("-----------feature rankings----------")
    print(feat_ranking)

    # run the algorithm on increasing features by adding the next most important feature
    for i in range(1,len(feat_ranking)):
        x_columns = X[feat_ranking[0:i]]
        # split the dataset into training and testing
        X_trn, X_tst, y_trn, y_tst = cross_v.train_test_split(x_columns, y, test_size=0.2, random_state=65)
        #----------------------------------------------------------------------------
        # train the model (supervised learning)
        model = svm.LinearSVC()
        model.fit(X_trn, np.ravel(y_trn))

        #----------------------------------------------------------------------------
        # test the model, print out the score and the features used
        print("-----------")
        print("-----------")
        print("  ")
        print(str(i+1) + " features score: " + str(model.score(X_tst, y_tst)))
        print("features used:")
        for feature in feat_ranking[0:i]:
            print(feature)

        print("  ")
    return

def fullAnalysis(agiFolder, nonAgiFolder):
    """Pulls the data from the text files in the folders, then trains and tests the data,
    performs the Recursive Feature Elimination, then runs the analysis described in the 
    previous function.
    
    Arguments:
        agiFolder {string} -- location of the folder that holds agitation-labelled data
        nonAgiFolder {string} -- location fo the folder that holds the other data
    """


    X, y = detectAgi(agiFolder, nonAgiFolder, classifier = "svm")
    ranking_list = audioRFE(X,y)
    RFE_Analysis(X,y,ranking_list)
    return 

