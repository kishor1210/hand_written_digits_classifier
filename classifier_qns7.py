
# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd

import itertools
# Import datasets, classifiers, and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)



# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data.shape
x_test = data[1500:] 
y_test = digits.target[1500:]



# Split data into 50% train and 50% test subsets
X_train, x_dev, y_train, y_dev = train_test_split(
            data[:1500], digits.target[:1500], test_size=0.25, shuffle=False
            )

# Create a classifier: a support vector classifier


# specify range of hyperparameters
# Set the parameters by cross-validation
gamma = [1e-2, 1e-3, 1e-4,1e-5]
C = [5,10,15]
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
print(results.head(20))


def stats(x):
    mn = x.min()
    mx = x.max()
    men = x.mean()
    md = x.median()
         
    return mn,mx,men,md


mn,mx,men,md = stats(results['train accuracy'])

print("Train accuracy stats:","Min :",mn,"Max :",mx, "Mean :",men,"Median :",md )


mn,mx,men,md = stats(results['dev accuracy'])

print("Dev accuracy stats:","Min :",mn,"Max :",mx, "Mean :",men,"Median :",md )

mn,mx,men,md = stats(results['test accuracy'])

print("Test accuracy stats:","Min :",mn,"Max :",mx, "Mean :",men,"Median :",md )



max_dev = results['dev accuracy'].max()
print("Best dev accuracy ")
print(results[results['dev accuracy']==max_dev].head(1))

