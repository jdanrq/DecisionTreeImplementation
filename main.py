from sklearn.tree import DecisionTreeClassifier as DTC
from pandas import DataFrame as df
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from src.decisionTree import decision_tree

###  Iris Dataset

dataset = load_iris()
X, Y = dataset['data'], dataset['target']
data_split = train_test_split(X, Y, test_size=0.2, random_state=2020)
Xtrain, Xtest, Ytrain, Ytest = data_split

model = decision_tree()

model.fit(Xtrain, Ytrain)
baseline = DTC(random_state=2019)
baseline.fit(Xtrain, Ytrain)

scores = [[model.score(Xtrain, Ytrain),    model.score(Xtest, Ytest)],
          [baseline.score(Xtrain, Ytrain), baseline.score(Xtest, Ytest)]]

results_iris = df(scores, columns=["train", "test"], index=["model", "baseline"])
print("Testing on Iris Dataset")
print(results_iris)

###  Wisconsin Breast Cancer Dataset

dataset = load_breast_cancer()
X, Y = dataset['data'], dataset['target']
data_split = train_test_split(X, Y, test_size=0.2, random_state=2020)
Xtrain, Xtest, Ytrain, Ytest = data_split

model = decision_tree()
model.fit(Xtrain, Ytrain)
baseline = DTC(random_state=2019)
baseline.fit(Xtrain, Ytrain)

scores = [[model.score(Xtrain, Ytrain),    model.score(Xtest, Ytest)],
          [baseline.score(Xtrain, Ytrain), baseline.score(Xtest, Ytest)]]

results_cancer = df(scores, columns=["train", "test"], index=["model", "baseline"])
print("Testing on Wisconsin Breast Cancer Dataset:")
print("results on the cancer dataset:")
print(results_cancer)

