#!/usr/bin/python3


from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()

index=[49,99,149]



#  Takeout data for training purpose
train_data = np.delete(iris.data , index , axis=0)
train_target = np.delete(iris.target , index )



#  Testing data
test_data = iris.data[index]
test_target = iris.target[index]




algo = tree.DecisionTreeClassifier()


# Model training
trained = algo.fit(train_data , train_target)

# tst = [iris_setosa_test_data .iris_versicolor_test_data ,]# iris_virginica_test_data]


# Predict output of applied testcase
resoutput = trained.predict(test_data)

print(resoutput)
print(test_target)
