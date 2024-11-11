import numpy as np
import cv2
import pickle as pkl
from load_data import load_data
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
import feature_extractor as fe
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from classifier_naive import classify_naive

def fit_classifier(X, y):

    # Split labels of [tile_type, num_crowns] into two separate arrays (one for each model we train)
    y_train_tiles = y[:,0]
    y_train_crowns = y[:,1]

    # train an AdaBoosted Random Forest Classifier for counting crowns
    est = RandomForestClassifier(random_state=0).fit(X_train, y_train_crowns)
    clf_crowns = AdaBoostClassifier(estimator=est, n_estimators=50, random_state=0).fit(X_train, y_train_crowns)

    # train a Logistic Regression Classifier for classifying tile type (using cross validation for fun)
    clf_tiles = LogisticRegressionCV(random_state=0, solver='lbfgs', max_iter=10000).fit(X_train, y_train_tiles)

    return clf_tiles, clf_crowns