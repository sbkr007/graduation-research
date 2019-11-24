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

def plot(columm_1_pd, graph_number):
    
    plt.figure(graph_number + 100)
    plt.plot(columm_1_pd) 
    plt.title('cluster' + str(graph_number) + 'mean')
    plt.xlabel('day') 
    plt.ylabel('stock price') 
    plt.grid(True)   
    plt.show() 

nikkei = pd.read_csv(str(998407) + '.csv', header=None).iloc[:,6].as_matrix()[::-1]
plot(nikkei, 100)
us = pd.read_csv('exchange.csv')['USD'].as_matrix()
plot(us, 200)

def derivative_plot(columm_1_pd, graph_number):
    
    derivative_price = {}   
    for i in range(1, len(columm_1_pd) - 1):
        derivative_price[i-1] = columm_1_pd[i] - columm_1_pd[i-1]              
    derivative_price = pd.Series(derivative_price)   
        
    plt.figure(graph_number + 100)
    plt.plot(derivative_price) 
    plt.title('cluster' + str(graph_number) + 'mean')
    plt.xlabel('day') 
    plt.ylabel('derivative price') 
    plt.grid(True)   
    plt.show()



def derivative_stock_price_plot(stock_code, graph_number):#figureにgraph_numberを入力  
        
    stock_code_price = pd.read_csv(str(stock_code) + '.csv', header=None).iloc[:,6]
       
    derivative_stock_code_price = {}   
    for i in range(1, len(stock_code_price) - 1):
        derivative_stock_code_price[i-1] = stock_code_price[i] - stock_code_price[i-1]              
    derivative_stock_code_price = pd.Series(derivative_stock_code_price)   
    
    plt.figure(graph_number)
    plt.plot(derivative_stock_code_price)   
    plt.title('cluster'+str(graph_number))
    plt.xlabel('day') 
    plt.ylabel('derivative price')
    plt.text(-1.7, derivative_stock_code_price[0], stock_code)
    plt.grid(True)   
    plt.show() 
    
    
def stock_price_plot(stock_code, graph_number):#figureにgraph_numberを入力   
     
    stock_code_price = pd.read_csv(str(stock_code) + '.csv', header=None).iloc[:,6] 
    
    plt.figure(graph_number)
    plt.plot(stock_code_price)
    
    plt.title('cluster'+str(graph_number))
    plt.xlabel('day') 
    plt.ylabel('close price')
    plt.text(-1.7, stock_code_price[0], stock_code)
    plt.grid(True) 
    plt.show() 


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
   
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    
    clf = [[] for i in range(4)] 
    predict_result = [[] for i in range(4)]  
    algorithm = [tree.DecisionTreeClassifier, RandomForestClassifier,
                 SVC, xgb.XGBClassifier]                    
    for j in range(0,2):
        clf[j] = algorithm[j](random_state=0)                            
        clf[j].fit(x_train,y_train) 
        predict_result[j] = clf[j].predict(x_test) 
    for j in range(2,4):
        clf[j] = algorithm[j](random_state=0)                            
        clf[j].fit(x_train_std,y_train) 
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

            
                                    
def predict_cluster(predict_model, learn_range):
    
    count = [0] * 4
    total = [0] * 4
    data = [[] for i in range(4)] 
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_number = int(cluster_data.iloc[[len(cluster_data)-1],[1]]
                     .as_matrix())
    nikkei_mean = pd.read_csv(str(998407) + '.csv', header=None).iloc[:,6] \
                  .as_matrix()[::-1]
    
    a = 9
    #for i in range(a, a+1):        
    for i in range(0, cluster_number + 1):
        each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
        stock_open = pd.DataFrame()
        stock_high = pd.DataFrame()
        stock_low = pd.DataFrame()
        stock_close = pd.DataFrame()               
        stock_volume = pd.DataFrame()
        
        for each_cluster_stock_code in each_cluster['stock_code']:                                                   
            stock_open_mean   = np.mean(pd.concat([stock_open,   
                pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None)
                .iloc[:,1]], axis=1), axis=1).as_matrix()[::-1]                
            stock_high_mean   = np.mean(pd.concat([stock_high,   
                pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None)
                .iloc[:,2]], axis=1), axis=1).as_matrix()[::-1]                
            stock_low_mean   = np.mean(pd.concat([stock_low,   
                pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None)
                .iloc[:,3]], axis=1), axis=1).as_matrix()[::-1]                
            stock_close_mean  = np.mean(pd.concat([stock_close,  
                pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None)
                .iloc[:,6]], axis=1), axis=1).as_matrix()[::-1]                
            stock_volume_mean = np.mean(pd.concat([stock_volume, 
                pd.read_csv(str(each_cluster_stock_code) + '.csv', header=None)
                .iloc[:,5]], axis=1), axis=1).as_matrix()[::-1]
            if i == 17:
                stock_price_plot(each_cluster_stock_code, i)
            #derivative_stock_price_plot(each_cluster_stock_code, i)
        
        #plot(pd.Series(stock_close_mean), i) 
        #derivative_plot(pd.Series(stock_close_mean), i)
        
        data = [stock_open_mean, 
                stock_high_mean, 
                stock_low_mean,
                stock_volume_mean, 
                nikkei_mean]   
        
        if predict_model == 1:
            group_data = [stock_close_mean] 
        if predict_model == 2:
            group_data = [stock_close_mean, data[3]] 
        if predict_model == 3:
            group_data = [stock_close_mean, data[0], data[3]] 
        if predict_model == 4:
            group_data = [stock_close_mean, data[0], data[1], data[2], data[3], data[4]] 
        
        predict_result = predict(group_data, learn_range) 
            
        high = (max([float(predict_result[0]), float(predict_result[1])]) 
                            / (float(predict_result[0]) + float(predict_result[1])))                                                                   
        for j in range(2,6):
            if predict_result[j] > high:      
                count[j-2]+= 1 
        for j in range(6,10):
            total[j-6]+= predict_result[j]
            
        #print i,high,predict_result 
                         
    return (round(float(count[0]) / float(cluster_number) * 100), 
            round(float(count[1]) / float(cluster_number) * 100), 
            round(float(count[2]) / float(cluster_number) * 100), 
            round(float(count[3]) / float(cluster_number) * 100), 
            total[0], total[1], total[2], total[3] )   

predict_cluster(1, 1)



def program_execution():
    color = ['blue', 'green', 'orange', 'red']
    algorithm = ["decision tree", "random forest", "SVM", "xgboost"]
    learn_range = 2
    predict_algorithm = 1
    data = np.array([predict_cluster(1, i)[predict_algorithm] for i in range(1, learn_range)])                                                                                     
    a = 0
    b = 0
    f = 0
    x = np.linspace(1, len(data), len(data))
    a, b = np.polyfit(x, data, 1) 
    f = a * x + b
    plt.scatter(x, data, color=color[predict_algorithm])
    plt.plot(x, f, color=color[predict_algorithm])
        
    plt.legend(loc='upper right')
    plt.xlabel("day") 
    plt.ylabel("predicted probability (%)")
    plt.grid(True)
    plt.show()
    
#program_execution()


end = time.time() 
print "経過時間", end - start   

"""
color = ['blue', 'green', 'orange', 'red']
for date in range(1, 3): 
    for predict_model in range(2, 3):     
        for predict_result in range(4, 8): 
            plt.scatter(date + predict_result * 0.05, 
                        predict_cluster(predict_model, date)[predict_result], 
                        color=color[predict_result - 4])    
plt.title("blue=decision,green=random,orange=SVM,red=boost")
plt.grid(True)
plt.show()
"""
        
        
        
        
        