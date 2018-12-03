from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
import math

import numpy as np

import pandas as pd

task_type = 0

dataset = pd.read_csv("./fea_labels.csv")

labels = dataset[["l1", "l2", "l3", "l4"]]

X = dataset.drop(columns=["l1", "l2", "l3", "l4"])
print(X.shape)
print(labels)
if task_type == 0:
    Y = labels.apply(lambda x: x.apply(lambda y: 1 if y > 0 else -1))
else:
    Y = labels

Y = Y.iloc[:, 0]

X = np.array(X)[:, 1:]
Y = np.array(Y)

# print(X.shape, Y.shape)
# data = np.hstack([X, Y.reshape([len(Y), 1])])
# print(data.shape)
cnt = 0
for train_idx, test_idx in KFold(n_splits=5).split(X):
    # print(train.shape)
    # print(test.shape)
    print("cv:", cnt)
    cnt += 1
    trainX, trainY, testX, testY = X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
    if task_type == 0:
        classifier = XGBClassifier()
        classifier.fit(trainX, trainY)
        y = classifier.predict(testX)
        cm = confusion_matrix(y, testY)
        true_num = sum(testY == 1)
        false_num = len(testY) - true_num
        print("true:", true_num, "false:", false_num)
        print("base acc = ", false_num / (true_num + false_num))
        print("our acc = ", (cm[0][0] + cm[1][1]) / sum(sum(cm)))
        print(classification_report(y, testY))
        print("balanced score = ", balanced_accuracy_score(y, testY))
        print(confusion_matrix(y, testY))
    else:
        regressor = XGBRegressor()
        regressor.fit(trainX, trainY)
        y = regressor.predict(testX)
        print("mse = ", mean_squared_error(y, testY))
        print("rmse = ", math.sqrt(mean_squared_error(y, testY)))




input()

print(X)
print(Y)

classifier = XGBClassifier()

classifier.fit(X, Y)

y = classifier.predict(X)

print(y)

print(confusion_matrix(y, Y))

# print(dataset)

labels = 0


