# -*- coding: utf-8 -*-
import jsm
import datetime
import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report

import time
start = time.time()


def predict(data, learn_range):
    x, y = [], []
    for start in range(len(data[0]) - learn_range):
        group_data = pd.DataFrame()
        for i in reversed(range(len(data))):
            add_data = pd.DataFrame(data[i])[start: start + learn_range]
            group_data = pd.concat((group_data, add_data), axis=0)
        group_data = group_data.as_matrix()

        if group_data[-1] < data[0][start + learn_range]:
            y.append(1)
        else:
            y.append(0)
        x.append([e for i in group_data for e in i])

    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    clf = [[] for i in range(4)]
    predict_result = [[] for i in range(4)]
    algorithm = [tree.DecisionTreeClassifier, RandomForestClassifier,
                 SVC, xgb.XGBClassifier]
    for j in range(0, 2):
        clf[j] = algorithm[j](random_state=0)
        clf[j].fit(x_train, y_train)
        predict_result[j] = clf[j].predict(x_test)
    for j in range(2, 4):
        clf[j] = algorithm[j](random_state=0)
        clf[j].fit(x_train_std, y_train)
        predict_result[j] = clf[j].predict(x_test_std)

    total = [[] for i in range(4)]
    for i in range(len(y_test)):
        for j in range(4):
            if predict_result[j][i] == 1:
                total[j].append(data[0][i + learn_range + len(y_train)]
                                - data[0][i + learn_range + len(y_train) - 1])
            else:
                total[j].append(data[0][i + learn_range + len(y_train) - 1]
                                - data[0][i + learn_range + len(y_train)])

    return (y.count(1), y.count(0),
            accuracy_score(predict_result[0], y_test),
            accuracy_score(predict_result[1], y_test),
            accuracy_score(predict_result[2], y_test),
            accuracy_score(predict_result[3], y_test),
            sum(total[0]), sum(total[1]), sum(total[2]), sum(total[3]))


def predict_cluster(predict_model, learn_range, predict_stock_code):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_number = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix())
    stock_cluster_number = 0
    predict_stock_cluster = []

    for i in range(0, cluster_number + 1):
        each_cluster = cluster_data[cluster_data['cluster_number'] == i]
        stock_close = pd.DataFrame()

        for stock_code in each_cluster['stock_code']:
            if stock_code == predict_stock_code:
                stock_cluster_number = i
            
            close_data = pd.read_csv(str(stock_code) + '.csv', header=None).iloc[:, 6]
            stock_close_mean = np.mean(pd.concat([stock_close, close_data], axis=1), axis=1).as_matrix()[::-1]
            
        predict_stock_cluster.append(stock_close_mean)

    if predict_model == 1:
        group_data = [pd.read_csv(str(predict_stock_code) + '.csv', header=None)
                      .iloc[:, 6].as_matrix()[::-1], predict_stock_cluster[stock_cluster_number]]

    return predict(group_data, learn_range)


def predict_only():
    predict_stock_code = 7203
    learn_range = 30
    predict_model = 1
    data = pd.read_csv(str(predict_stock_code) + '.csv', header=None).iloc[:, 6].as_matrix()[::-1]
    original_predict = np.array([predict([data], i)[2] for i in range(1, learn_range)])
    a1 = 0
    b1 = 0
    f1 = 0
    x1 = np.linspace(1, len(original_predict), len(original_predict))
    a1, b1 = np.polyfit(x1, original_predict, 1)
    f1 = a1 * x1 + b1
    plt.plot(x1, f1, color='red')

    combine_predict = np.array([predict_cluster(predict_model, i, predict_stock_code)[2]
                                for i in range(1, learn_range)])
    a2 = 0
    b2 = 0
    f2 = 0
    x2 = np.linspace(1, len(combine_predict), len(combine_predict))
    a2, b2 = np.polyfit(x2, combine_predict, 1)
    f2 = a2 * x2 + b2
    plt.plot(x2, f2, color='blue')

    high = 0
    low = 0
    for i in range(1, learn_range):
        if (predict([data], i)[2]
                < predict_cluster(predict_model, i, predict_stock_code)[2]):
            high += 1

        if (predict([data], i)[2]
                > predict_cluster(predict_model, i, predict_stock_code)[2]):
            low += 1

        plt.scatter(i, predict_cluster(predict_model, i,
                                       predict_stock_code)[2], color='blue')
        plt.scatter(i, predict([data], i)[2], color='red')
        if (predict([data], i)[2]
                == predict_cluster(predict_model, i, predict_stock_code)[2]):
            plt.scatter(i, predict([data], i)[2], color='0.6')

    if high == 0 and low != 0:
        print '0'
    if high != 0 and low == 0:
        print '1'
    if high == 0 and low == 0:
        print '0'
    else:
        print float(high) / (float(high) + float(low))

    plt.grid(True)
    plt.show()


predict_only()


end = time.time()
print "経過時間", end - start
