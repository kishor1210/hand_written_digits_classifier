# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:13:36 2022

@author: kishor
"""

# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import itertools
# Import datasets, classifiers, and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import rescale, resize

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
                    

data = np.array([resize(digits.images[i].reshape(8,8),(4,4)) for i in range(len(digits.images))])
#data = np.array([resize(digits.images[i].reshape(8,8),(16,16)) for i in range(len(digits.images))])

#data = np.array([resize(digits.images[i].reshape(8,8),(32,32)) for i in range(len(digits.images))])

print(data.shape)



 # flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
 


x_test = data[1600:] 
y_test = digits.target[1600:]





# Split data into 50% train and 50% test subsets
X_train, x_dev, y_train, y_dev = train_test_split(
            data[:1600], digits.target[:1600], test_size=0.3, shuffle=False
            )

# Create a classifier: a support vector classifier


# specify range of hyperparameters
# Set the parameters by cross-validation
gamma = [1e-2, 1e-3, 1e-4]
C = [5,10]
list = [gamma,C]
combinations = [p for p in itertools.product(*list)]


hyper_params,train_acc,dev_acc,test_acc=[],[],[],[]
for gamma,c in combinations:
    hyper_params.append((gamma,c))
    # specify model
    model = svm.SVC( kernel='rbf',gamma=gamma,C=c)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train,model.predict(X_train)))
    dev_acc.append(accuracy_score(y_dev,model.predict(x_dev)))
    test_acc.append(accuracy_score(y_test,model.predict(x_test)))
   



results = pd.DataFrame({'hyper_params':hyper_params,'train accuracy':train_acc,'dev accuracy':dev_acc,'test accuracy':test_acc}) 
    


print(results.head(10))

max_dev = results['dev accuracy'].max()

print("Best dev accuracy ")
print(results[results['dev accuracy']==max_dev].head(1))


