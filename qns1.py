# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:37:33 2022

@author: kishor
"""

import pandas as pd
import numpy as np
import itertools
import unittest
import argparse
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from skimage.transform import *

seed = 42

np.random.seed(seed)



digits = datasets.load_digits()

n_samples = len(digits.images)
img = digits.images.reshape((n_samples, -1))
print(img.shape)

xtrain, x_test, ytrain, y_test = train_test_split(img, digits.target, test_size=0.15, stratify=digits.target, random_state=seed)
print(xtrain.shape,x_test.shape)

xtrain, x_dev, ytrain, y_dev = train_test_split(xtrain, ytrain, test_size=0.15, stratify=ytrain, random_state=seed)
print(xtrain.shape,x_test.shape)
print(x_dev.shape,y_dev.shape)

#for testing purpose
xtrain2, x_test2, ytrain2, y_test2 = train_test_split(img, digits.target, test_size=0.15, stratify=digits.target, random_state=seed)
xtrain2, x_dev2, ytrain2, y_dev2 = train_test_split(xtrain2, ytrain2, test_size=0.15, stratify=ytrain2, random_state=seed)


get_seed = np.random.get_state()[1][0]

def testing(x,y):
    return np.array_equal(x, y, equal_nan=True)

class TestNotebook(unittest.TestCase):
    
    def test_random_seed_same(self):
        self.assertTrue(testing(xtrain,xtrain2))
        
    def test_random_seed_not_same(self):
        if testing(xtrain,xtrain2)==True:
            a = False
        else:
            b = True
        self.assertFalse(False)
        
        
unittest.main(argv=[''], verbosity=2, exit=False)