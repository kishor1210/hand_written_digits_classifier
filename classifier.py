# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:35:51 2022

@author: kishor
"""

import pandas as pd
import numpy as np
import argparse
	# Standard scientific Python imports
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
#load the digits
digits = load_digits()




def build_classifier(args):

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    random_state = args.random_state

    X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.2, shuffle=False,random_state=random_state
            )

    X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False,random_state=random_state)

        
    if args.clf_name=='svm':
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10], 
                          'gamma': [1, 0.1, 0.01],
                          'kernel': ['rbf']} 
              
        model = GridSearchCV(SVC(), param_grid, refit = True)
    else:
        param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 6)}
            
        model = GridSearchCV(DecisionTreeClassifier(), param_grid,refit = True)
        
    model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test,y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

        
    return precision,recall,accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_name', default='svm') 
    parser.add_argument('--random_state' , default= 34) 
    args = parser.parse_args()
    
    
    precision,recall,accuracy = build_classifier(args)
    
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)





