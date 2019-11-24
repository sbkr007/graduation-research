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

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

import time
start = time.time() 


def predict(data, learn_range):
    x, y = [], []
    for start in range(len(data[0]) - learn_range):
        group_data = pd.DataFrame()
        
        for i in reversed(range(len(data))):
            add_data = pd.DataFrame(data[i])[start : start + learn_range]
            group_data = pd.concat((group_data, add_data), axis=0)
            
        group_data = group_data.as_matrix()

        if group_data[-1] < data[0][start + learn_range]:
            y.append(1)
        else:
            y.append(0)
            
        x.append([e for i in group_data for e in i])

    x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)
    
    select = SelectFromModel(RandomForestClassifier(), threshold="median").fit(x_train, y_train)
    X_train_selected = select.transform(x_train)
    X_test_selected = select.transform(x_test) 
       
    algorithm = RandomForestClassifier
    
    predict_result = [] 
    predict_result2 = []        
    if algorithm == tree.DecisionTreeClassifier or algorithm == RandomForestClassifier:
        clf = algorithm(random_state=0)
        clf.fit(x_train, y_train)
        predict_result = clf.predict(x_test)
        
        clf2 = algorithm(random_state=0)
        clf2.fit(X_train_selected, y_train)
        predict_result2 = clf2.predict(X_test_selected)
        
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
        before = data[0][i + learn_range + len(y_train) - 1]
        after = data[0][i + learn_range + len(y_train)]
        if predict_result[i] == 1:
            total.append(after - before)
        else:
            total.append(before - after)
    
    count_0 = float(y.count(0))                 
    count_1 = float(y.count(1))                            
    high = max([count_0, count_1]) / (count_0 + count_1)
    
    print accuracy_score(predict_result, y_test), accuracy_score(predict_result2, y_test)
    #print clf.feature_importances_
    #print clf2.feature_importances_
    print select.get_support()
    
    return (count_0, count_1, high,
            accuracy_score(predict_result, y_test),
            sum(total),
            clf.feature_importances_ ,
            accuracy_score(predict_result2, y_test))
                   
                                
def predict_cluster(predict_model, learn_range):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_number = int(cluster_data.iloc[[len(cluster_data)-1],[1]].as_matrix())
    exchange = pd.read_csv('exchange.csv')
    indicator = pd.read_csv('indicator.csv')
    count = 0                
    total = []   
    importance = [[] for i in range(5)]
    cluster_count = 0
     
    for i in range(0, cluster_number + 1):
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1:
            print i
            cluster_count+= 1   
            each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
            stock_close = pd.DataFrame()
            stock_volume = pd.DataFrame()                
                    
            for stock_code in each_cluster['stock_code']:  
                stock_data = pd.read_csv(str(stock_code) + '.csv', header=None)                                                                                   
                stock_close_mean = np.mean(pd.concat([stock_close, stock_data.iloc[:,6]], axis=1), axis=1).as_matrix()[::-1]                                                   
                stock_close_combine = pd.concat([stock_close, stock_data.iloc[:,6]], axis=1).as_matrix()[::-1]  
                stock_volume_mean = np.mean(pd.concat([stock_volume, stock_data.iloc[:,5]], axis=1), axis=1).as_matrix()[::-1]                                                   
                stock_volume_combine = pd.concat([stock_volume, stock_data.iloc[:,5]], axis=1).as_matrix()[::-1]  
                
            stock_close_combine = [e for i in stock_close_combine for e in i]
            
            if predict_model == 1:
                group_data = [stock_close_mean]             
            if predict_model == 2:
                group_data = [stock_close_mean, stock_close_combine] 
            if predict_model == 3:
                group_data = [stock_close_mean, stock_volume_mean] 
            if predict_model == 4:
                group_data = [stock_close_mean, 
                            stock_close_combine,
                            exchange['USD'].as_matrix(),
                            indicator['nikkei'].as_matrix(),
                            indicator['topix'].as_matrix()
                            ]   
            if predict_model == 5:
                group_data = [stock_close_mean, 
                            stock_close_combine,
                            exchange['USD'].as_matrix(),
                            indicator['nikkei'].as_matrix(),
                            indicator['topix'].as_matrix(),
                            indicator['s_p_500'].fillna(method='ffill').fillna(method='bfill').as_matrix(),
                            indicator['dow'].fillna(method='ffill').fillna(method='bfill').as_matrix(), 
                            indicator['hang_seng_index'].fillna(method='ffill').fillna(method='bfill').as_matrix(), 
                            indicator['cac_40'].fillna(method='ffill').fillna(method='bfill').as_matrix()                  
                            ] 
            
            predict_result = predict(group_data, learn_range) 
                
            if predict_result[3] > predict_result[2]:                  
                total.append(float(predict_result[4]))
                count+= 1
                
            #for a in range(5):
                #importance[a].append(predict_result[5][a])
        
    #for b in range(9):
         #print sum(importance[b])
                                
    return sum(total), round(float(count) / float(cluster_count) * 100) 

print predict_cluster(5, 1)  



end = time.time() 
print "経過時間", end - start   