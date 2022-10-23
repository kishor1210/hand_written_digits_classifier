# -*- coding: utf-8 -*-
"""
Created on Thu OCT 23 11:22:51 2022

@author: Kishor
"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers, and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False
            )

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
 )


def completely_biased_y_n(x):
    return pd.value_counts(x).shape[0]
def preds_all_classes_y_n(x):
    return pd.value_counts(x).shape[0]


import unittest

class TestNotebook(unittest.TestCase):
    
    def test_completely_biased_y_n(self):
        self.assertEqual(completely_biased_y_n(predicted), 10)
        
    def test_preds_all_classes_y_n(self):
        self.assertEqual(preds_all_classes_y_n(predicted), 10)

unittest.main(argv=[''], verbosity=2, exit=False)
