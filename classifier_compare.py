
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 23:32:54 2022

@author: kishor
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier

#load the digits
digits = load_digits()

# compare with labels 
def label_compare(y_true,y_pred):
    cnt=0
    
    for x,y in zip(y_true,y_pred):
        if x==y:
            cnt+=1
    return cnt


def build_classifier(digits,classifier):
    acc =[]
    TP=[]
    precision, recall = [],[]
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    target = digits.target
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    for train_index, test_index in skf.split(data, target):
        x_train_fold, x_test_fold = data[train_index], data[test_index]
        y_train_fold, y_test_fold = target[train_index], target[test_index]
        
        if classifier == "svm":
            # defining parameter range
            param_grid = {'C': [0.1, 1, 10], 
                          'gamma': [1, 0.1, 0.01],
                          'kernel': ['rbf']} 
              
            model = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
        else:
            param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 6)}
            
            model = GridSearchCV(DecisionTreeClassifier(), param_grid,refit = True, verbose = 3)
        
        model.fit(x_train_fold, y_train_fold)
        
        y_pred = model.predict(x_test_fold)
        
        TP.append(label_compare(y_test_fold,y_pred))
        precision.append(precision_score(y_test_fold, y_pred, average='weighted'))
        recall.append(recall_score(y_test_fold,y_pred, average='weighted'))
        acc.append(accuracy_score(y_test_fold, y_pred))
        
    return acc,precision,recall,TP

final_df = pd.DataFrame()

final_df["svm_acc"],final_df["svm_precision"],final_df["svm_recall"],final_df["svm_TP"] = build_classifier(digits,'svm')

final_df["dtree_acc"],final_df["dtree_precision"],final_df["dtree_recall"],final_df["dtree_TP"]  = build_classifier(digits,'dtree')


print("mean :", final_df["svm_acc"].mean(),  final_df["dtree_acc"].mean())    

print("std :", final_df["svm_acc"].std(), final_df["dtree_acc"].std() )    

