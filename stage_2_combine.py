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


def predict2(data, learn_range):
    x = []
    y = []
    z = []
    xz = []
    
    for start in range(len(data[0]) - learn_range):
        group_data = pd.DataFrame()
        
        for i in reversed(range(len(data))):
            add_data = pd.DataFrame(data[i])[start: start + learn_range]
            group_data = pd.concat((group_data, add_data), axis=0)
            
        group_data = group_data.as_matrix()
                                             
        if ((data[0][start + learn_range] - group_data[-1]) / group_data[-1] <= 0.01 and 
            (data[0][start + learn_range] - group_data[-1]) / group_data[-1] >= -0.01):
            z.append(0)
            xz.append([e for i in group_data for e in i] + [0])
        else:
            z.append(2) 
            xz.append([e for i in group_data for e in i] + [2])
                                                       
        if group_data[-1] < data[0][start + learn_range]:
            y.append(1)
        else:
            y.append(-1)   
        
        x.append([e for i in group_data for e in i])
    
    x_train, x_test, z_train, z_test, y_train, y_test, xz_train, xz_test \
        = train_test_split(x, z, y, xz, test_size=0.3, random_state=0, shuffle=False)
        
    algorithm = RandomForestClassifier
    
    clf = [[] for i in range(3)]
    for i in range(3):    
        clf[i] = algorithm(random_state=0)
    
    xz2 = []
    xz2y = []
    xz0 = []
    xz0y = []
    for i in range(len(y_train)):
        if xz_train[i][-1] == 2:
            xz2y.append(y_train[i])
            xz2.append(xz_train[i])             
        else:
            xz0y.append(y_train[i])
            xz0.append(xz_train[i])  
    
    clf[0].fit(x_train, z_train)        
    clf[1].fit(xz2, xz2y)
    clf[2].fit(xz0, xz0y)
     
    predict_result = [[] for i in range(2)]
    
    predict_result[0] = clf[0].predict(x_test)
       
    for i in range(len(predict_result[0])):
        if predict_result[0][i] == 2:
            predict_result[1].append(clf[1].predict([x_test[i] + [2]]))
        else:
            predict_result[1].append(clf[2].predict([x_test[i] + [0]]))
        
    high = (max([float(y.count(1)), float(y.count(-1))]) 
            / (float(y.count(1)) + float(y.count(-1)))) 
      
    return (y.count(1), y.count(-1), high,
            accuracy_score(predict_result[1], y_test))


def predict_cluster2(predict_model, learn_range):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_number = int(cluster_data.iloc[[len(cluster_data)-1],[1]].as_matrix())
    count = 0
                         
    for i in range(0, cluster_number + 1):
        each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
        stock_close = pd.DataFrame()               
                
        for each_cluster_stock_code in each_cluster['stock_code']:                                                                          
            close_data = pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None).iloc[:,6]                                                                         
            stock_close_mean = np.mean(pd.concat([stock_close, close_data], axis=1), axis=1).as_matrix()[::-1]   
                
            if predict_model == 1:
                group_data = [stock_close_mean]
        
        predict_result = predict2(group_data, learn_range) 
                                                                                      
        if predict_result[3] > predict_result[2]:                  
            #print i, predict_result[3], predict_result[2] 
            count+= 1
            
    return round(float(count) / float(cluster_number) * 100)
    

def program_execution():
    learn_range = 30
    data = np.array([predict_cluster2(1, i) for i in range(1, learn_range)])                                                                                     
    a = 0
    b = 0
    f = 0
    x = np.linspace(1, len(data), len(data))
    a, b = np.polyfit(x, data, 1) 
    f = a * x + b
    plt.scatter(x, data, color="b")
    plt.plot(x, f, color="b")
    plt.grid(True)
    plt.show() 
    
program_execution()

            
end = time.time() 
print "経過時間", end - start  
