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
        
        if (data[0][start + learn_range] - group_data[-1]) / group_data[-1] > 0.01:
            y.append(1)
        if (data[0][start + learn_range] - group_data[-1]) / group_data[-1] < -0.01:
            y.append(-1)    
        if ((data[0][start + learn_range] - group_data[-1]) / group_data[-1] <= 0.01 and 
            (data[0][start + learn_range] - group_data[-1]) / group_data[-1] >= -0.01):
            y.append(0)
            
        x.append([e for i in group_data for e in i])
    
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)
   
    algorithm = RandomForestClassifier
    
    predict_result = []        
    if algorithm == tree.DecisionTreeClassifier or algorithm == RandomForestClassifier:
        clf = algorithm(random_state=0)
        clf.fit(x_train, y_train)
        predict_result = clf.predict(x_test)
        
    if algorithm == SVC or algorithm == xgb.XGBClassifier:
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)
        clf = algorithm(random_state=0)
        clf.fit(x_train_std, y_train)
        predict_result = clf.predict(x_test_std)
    
    total = []
    for i in range(len(y_test)):
        if predict_result[i] == 1:
            total.append(data[0][i + learn_range + len(y_train)]
                         - data[0][i + learn_range + len(y_train) - 1])
        if predict_result[i] == -1:
            total.append(data[0][i + learn_range + len(y_train) - 1]
                         - data[0][i + learn_range + len(y_train)])

    return (y.count(1), y.count(0), y.count(-1),
            accuracy_score(predict_result, y_test),
            sum(total))


def predict_cluster(predict_model, learn_range):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_number = int(cluster_data.iloc[[len(cluster_data)-1],[1]].as_matrix())
    total = []
    exchange = pd.read_csv('exchange.csv')
    count = 0
                         
    for i in range(0, cluster_number + 1):
        each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
        stock_close = pd.DataFrame()               
                
        for each_cluster_stock_code in each_cluster['stock_code']:                                                                          
            close_data = pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None).iloc[:,6]                                                                         
            stock_close_mean = np.mean(pd.concat([stock_close, close_data], axis=1), axis=1).as_matrix()[::-1]   
        """
        derivative_price = []    
        for i in range(1, len(stock_close_mean) - 1):
            derivative_price.append((stock_close_mean[i] - stock_close_mean[i-1]) / stock_close_mean[i-1])
        x = np.linspace(1, len(derivative_price), len(derivative_price))
        plt.plot(x, derivative_price, color="b")
        plt.grid(True)
        plt.show()
        """    
        if predict_model == 1:
            group_data = [stock_close_mean]
            
        if predict_model == 2:
            group_data = [stock_close_mean, exchange['PHP'].as_matrix()] 
        if predict_model == 3:
            group_data = [stock_close_mean, exchange['SGD'].as_matrix()]
        if predict_model == 4:
            group_data = [stock_close_mean, exchange['THB'].as_matrix()]
        if predict_model == 5:
            group_data = [stock_close_mean, exchange['KWD'].as_matrix()]
        if predict_model == 6:
            group_data = [stock_close_mean, exchange['SAR'].as_matrix()] 
        
        predict_result = predict(group_data, learn_range) 
            
        high = (max([float(predict_result[0]), float(predict_result[2])]) 
                            / (float(predict_result[0]) + float(predict_result[2])))                                                                   
        
        if predict_result[3] > high:                  
            total.append(float(predict_result[4]))
            count+= 1
           
    return sum(total), round(float(count) / float(cluster_number) * 100)
    

def program_execution():

    learn_range = 30
    predict_model = 1      
    data = np.array([predict_cluster(predict_model, i)[1] for i in range(1, learn_range)])                                                                                     
    a = 0
    b = 0
    f = 0
    x = np.linspace(1, len(data), len(data))
    a, b = np.polyfit(x, data, 1) 
    f = a * x + b
    plt.scatter(x, data, color="r")
    plt.plot(x, f, color="r")
    plt.grid(True)
    plt.show()
    """
    for i in range(1, learn_range):
        predict_cluster(predict_model, i)
    """    
program_execution()

"""
def program_execution():
    learn_range = 15
    for i in range(2, 7):
        predict_model = i
        data = np.array([predict_cluster2(predict_model, i)[0] for i in range(2, learn_range)])
        x = np.linspace(1, len(data), len(data)) 
        a = 0
        b = 0
        a, b = np.polyfit(x, data, 1) 
        f = 0
        f = a * x + b
        if predict_model == 1:
            plt.scatter(x, data, color="blue")
            plt.plot(x, f, color="blue")
        if predict_model == 2:
            plt.scatter(x, data, color="red")
            plt.plot(x, f, color="red")
        if predict_model == 3:
            plt.scatter(x, data, color="yellow")
            plt.plot(x, f, color="yellow")
        if predict_model == 4:
            plt.scatter(x, data, color="black")
            plt.plot(x, f, color="black")
        if predict_model == 5:
            plt.scatter(x, data, color="green")
            plt.plot(x, f, color="green")
        if predict_model == 6:
            plt.scatter(x, data, color="purple")
            plt.plot(x, f, color="purple")
        plt.grid(True)
        plt.show()
 
program_execution()
""" 

end = time.time() 
print "経過時間", end - start  
