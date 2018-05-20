#!/usr/bin/python3

from sklearn import tree

features = [
			[110,0],
			[120,0],
			[130,1],
			[140,1]
		]

output = ["Apple",'Apple','Orange','Orange']


algo = tree.DecisionTreeClassifier()

trained = algo.fit(features,output)

resoutput = trained.predict([[127,0]])
print(resoutput)