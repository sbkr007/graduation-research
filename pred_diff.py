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

def pred(data, term):
    x, y = [], []
    for start in range(len(data[0]) - term):
        group_data = pd.DataFrame()
        
        for i in reversed(range(len(data))):
            add_data = pd.DataFrame(data[i])[start : start+term]
            group_data = pd.concat((group_data, add_data), axis=0)
            
        group_data = group_data.as_matrix()

        if group_data[-1] < data[0][start + term]:
            y.append(1)
        else:
            y.append(0)
            
        x.append([e for i in group_data for e in i])
      
    train_size_adjust = (len(data[0])*0.7 - float(term)) / float((len(data[0]) - term)) 
     
    x_train, x_test, y_train, y_test \
    = train_test_split(x, y, train_size=train_size_adjust, random_state=0, shuffle=False)
     
    algorithm = tree.DecisionTreeClassifier
    
    pred_result = []        
    if (algorithm == tree.DecisionTreeClassifier or 
        algorithm == RandomForestClassifier):
        clf = algorithm(random_state=0)
        clf.fit(x_train, y_train)
        pred_result = clf.predict(x_test)
        
    if algorithm == SVC or algorithm == xgb.XGBClassifier:
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)
        clf = algorithm(random_state=0)
        clf.fit(x_train_std, y_train)
        pred_result = clf.predict(x_test_std)
    
    if y_train.count(0) > y_train.count(1):         
        high = float(y_test.count(0)) / (float(y_test.count(0)) + float(y_test.count(1)))
    if y_train.count(0) <= y_train.count(1): 
        high = float(y_test.count(1)) / (float(y_test.count(0)) + float(y_test.count(1)))
                                       
    return (y.count(1), y.count(0),
            accuracy_score(pred_result, y_test),
            high)


def pred_cluster(model, term, code):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    row =  cluster_data[cluster_data['stock_code'] == code]
    cluster_num = int(row['cluster_number'].as_matrix())  
    pred_cluster = cluster_data[cluster_data['cluster_number'] == cluster_num]
    close_all = pd.DataFrame()
    
    for stock_code in pred_cluster['stock_code']:
        close = pd.read_csv(str(stock_code) + '.csv', header=None).iloc[:, 6][::-1]
        close_all = pd.concat([close_all, close], axis=1)
           
    close_mean = np.mean(close_all, axis=1).as_matrix()
    
    data = pd.read_csv(str(code) + '.csv', header=None)[::-1]        
    pred_data = data.iloc[:, 6].as_matrix() 
                
    if model == 1:
        group_data = [pred_data, close_mean]
        
    if model == 2:
        diff = [pred_data[i] - close_mean[i] for i in range(len(pred_data))]  
        group_data = [pred_data, diff]
    
    if model == 3:
        diff = [pred_data[i] - close_mean[i] for i in range(len(pred_data))]
        indicator = pd.read_csv('indicator.csv')
        exchange = pd.read_csv('exchange.csv')
        group_data = [pred_data, diff, indicator['nikkei'].as_matrix(), exchange['USD'].as_matrix()]    
    
    if model == 4:
        diff = [pred_data[i] - close_mean[i] for i in range(len(pred_data))]
        diff_diff = pd.DataFrame(diff).diff().dropna().as_matrix() 
        pred_data_short = pd.DataFrame(pred_data).drop(0).as_matrix()
        group_data = [pred_data_short, diff_diff]
                
    return pred(group_data, term)
         

def csv_plot_origin_cluster(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    n = map(int, np.linspace(start, term-1, term-1))
    a = 1
               
    for i in range(0, 20):        
    #for i in range(0, cluster_num + 1):            
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
            diff = [[] for c in range((term-1))] 
            
            for code in each_cluster['stock_code']: 
                data = (pd.read_csv(str(code) + '.csv', header=None).iloc[:, 6].as_matrix()[::-1]) 
                row = pd.DataFrame() 
                            
                for j in range(start, term):  
                    if a == 1:                   
                        diff[j-1].append(pred_cluster(model, j, code)[2])
                    if a == 2:
                        diff[j-1].append(pred_cluster(model, j, code)[3])
                    if a == 3:
                        diff[j-1].append(pred([data], j)[2])
                    if a == 4:
                        diff[j-1].append(pred([data], j)[3])
                    row = pd.concat([row, pd.DataFrame(diff[j-1])], axis=1)  
                             
            result = pd.DataFrame(row.as_matrix(), columns=n)
            if a == 1:
                result.to_csv('model' + str(model) + '_' + str(i) + '.csv', index=None)
            if a == 2:
                result.to_csv('model_chance' + str(i) + '.csv', index=None)
            if a == 3:
                result.to_csv('origin_' + str(i) + '.csv', index=None)  
            if a == 4:
                result.to_csv('origin_chance_' + str(i) + '.csv', index=None)        
                     
#csv_plot_origin_cluster(1, 31, 2) 

def plot_cluster_csv(model):
    
    for cluster in range(0,20):
        d = [[] for i in range(31)]
        if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
        for term in range(1, 31):
            a = pd.read_csv('model' + str(model) + '_' + str(cluster) + '.csv').iloc[:, term-1]
            b = pd.read_csv('model_chance_' + str(cluster) + '.csv').iloc[:, term-1]
            c = pd.read_csv('origin_' + str(cluster) + '.csv').iloc[:, term-1]
            d[term-1] = (a - b) * 100
                        
        plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                     d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                     d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], 
                     d[25], d[26], d[27], d[28], d[29]]
        
        fig = plt.figure()    
        ax1 = fig.add_subplot(111)    
        ax1.boxplot(plot_data)
        #ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]) 
        ax1.grid()
        plt.title('cluster'+str(cluster))
        plt.legend(loc='upper right')
        plt.xlabel("term") 
        plt.ylabel("predicted probability(%)")
        plt.show()
        

#plot_cluster_csv(2)    

def plot_term_csv(model):    
    term = 5
    #for term in range(term, term+1):
    for term in range(1, 31):
        d = [[] for i in range(20)]    
        for cluster in range(0,20):
            if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
            a = pd.read_csv('model' + str(model) + '_' + str(cluster) + '.csv').iloc[:, term-1]
            #b = pd.read_csv('model_chance_' + str(cluster) + '.csv').iloc[:, term-1]
            c = pd.read_csv('origin_' + str(cluster) + '.csv').iloc[:, term-1]
            d[cluster] = (a - c) * 100
        
        plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                 
        fig = plt.figure()    
        ax1 = fig.add_subplot(111)    
        ax1.boxplot(plot_data)
        ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
        ax1.grid()
        plt.title('term'+str(term))
        plt.legend(loc='upper right')
        plt.xlabel("cluster") 
        plt.ylabel("predicted probability")
        plt.show()
                            
#plot_term_csv(3)


def plot_csv_all(model):    
    d = [[] for o in range(20)]     
    for term in range(1, 31):    
        for cluster in range(20):
            if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
            a = pd.read_csv('model' + str(model) + '_' + str(cluster) + '.csv').iloc[:, term-1]
            #b = pd.read_csv('model_chance_' + str(cluster) + '.csv').iloc[:, term-1]
            c = pd.read_csv('origin_' + str(cluster) + '.csv').iloc[:, term-1]
            d[cluster].append(((a - c) * 100))
            
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                 
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data)
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
    ax1.grid()
    plt.title('model'+str(model))
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability(%)")
    plt.show()
    
plot_csv_all(2) 


            
end = time.time() 
print "経過時間", end - start    

  
"""    
def plot_cluster(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    o = [[] for i in range(cluster_num + 1)]
    d = [[] for i in range(cluster_num + 1)]
        
    for i in range(0, cluster_num + 1):        
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i]                 
            for code in each_cluster['stock_code']: 
                data = (pd.read_csv(str(code) + '.csv', header=None)
                        .iloc[:, 6].as_matrix()[::-1])                
                for j in range(start, term):  
                    d[i].append(pred_cluster(model, j, code)[2])           
                    o[i].append(pred([data], j)[2])
                           
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
    
    plot_data2 = [o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], 
                  o[9], o[10], o[11], o[12], o[13], o[14], o[15], o[16], 
                  o[17], o[18], o[19]]
                                           
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 1, 1)    
    ax1.boxplot(plot_data) 
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax1.grid()
    
    ax2 = fig.add_subplot(2, 1, 2)    
    ax2.boxplot(plot_data2) 
    ax2.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax2.grid()
    
    plt.show()

def regression(code, start, term, model):
    if model == 1:
        c = 'g'
    else:
        c = 'b'
        
    data = pd.read_csv(str(code) + '.csv', header=None)[::-1]        
    price = data.iloc[:, 6].as_matrix()
           
    original = np.array([pred([price], i)[2] for i in range(start, term)])    
    a1, b1, f1 = 0, 0, 0
    x1 = np.linspace(1, len(original), len(original))
    a1, b1 = np.polyfit(x1, original, 1)
    f1 = a1 * x1 + b1
    plt.plot(x1, f1, color='r')

    combine = np.array([pred_cluster(model, i, code)[2]
                                for i in range(start, term)])
    a2, b2, f2 = 0, 0, 0
    x2 = np.linspace(1, len(combine), len(combine))
    a2, b2 = np.polyfit(x2, combine, 1)
    f2 = a2 * x2 + b2
    plt.plot(x2, f2, color=c)
    
    for i in range(start, term):
        plt.scatter(i, pred([price], i)[2], color='r')
        plt.scatter(i, pred_cluster(model, i, code)[2], color=c)
        
        if pred([price], i)[2] == pred_cluster(model, i, code)[2]:
            plt.scatter(i, pred([price], i)[2], color='0.6')

    plt.grid(True)
    plt.show()
    
#regression(7203, 1, 30, 2)

def boxplot(code, start, term, model):
    data = pd.read_csv(str(code) + '.csv', header=None)[::-1]        
    close = data.iloc[:, 6].as_matrix()   
    original = [pred([close], i)[2] for i in range(start, term)]       
    combine = [pred_cluster(model, i, code)[2] for i in range(start, term)]
    
    plot_data = [original, combine]
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.boxplot(plot_data) 
    ax.grid()
    plt.show()
    
#boxplot(7203, 1, 30, 2)

def plot_data(start, term, model):
    stock_code = pd.read_csv('nikkei_stock_code_2.csv')
    original = []
    combine = []
    for code in stock_code['stock_code']:
        data = (pd.read_csv(str(code) + '.csv', header=None)
            .iloc[:, 6].as_matrix()[::-1])
        
        original_pred = [pred([data], i)[2] for i in range(start, term)]
        original.append(original_pred)
        
        combine_pred = [pred_cluster(model, i, code)[2] for i in range(start, term)]
        combine.append(combine_pred)
        
        if code == 1605:
            break
         
    plot_model = 1 
        
    if plot_model == 1:        
        plot_data = [original, combine]
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        ax.boxplot(plot_data) 
        ax.grid()
        plt.show()
    else:    
        plot_data = [original, combine]
        plt.violinplot(plot_data, showmedians=True)
        plt.show()
      
#plot_data(5, 6, 2)

def plot_term(start, term, model):
    stock_code = pd.read_csv('nikkei_stock_code_2.csv')
    d = [[] for i in range(term)]
    for code in stock_code['stock_code']:
        data = (pd.read_csv(str(code) + '.csv', header=None)
                .iloc[:, 6].as_matrix()[::-1])
        
        for i in range(start, term):             
            #d[i].append(pred([data], i)[2])
            d[i].append(pred_cluster(model, i, code)[2])
                
        #if code == 1605:
            #break
            
    plot_data = [d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10],
                 d[11], d[12], d[13], d[14], d[15], d[15], d[17], d[18], d[19],
                 d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28]
                 , d[29]]
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    ax.boxplot(plot_data) 
    ax.grid()
    plt.show()

#plot_term(1, 30, 2)

def plot_cluster(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    d = [[] for i in range(cluster_num + 1)]
    
    a = 12       
    #for i in range(a, a+1):        
    for i in range(0, cluster_num + 1):         
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i]                 
            for code in each_cluster['stock_code']: 
                data = (pd.read_csv(str(code) + '.csv', header=None)
                        .iloc[:, 6].as_matrix()[::-1])                
                for j in range(start, term):  
                    d[i].append(pred_cluster(model, j, code)[2]) 
                    #d[i].append(pred([data], j)[2])          
                           
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                                               
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data) 
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax1.grid()
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability (%)")
    plt.show()

#plot_cluster(1, 30, 2) 

def plot_origin_cluster(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    d = [[] for i in range(cluster_num + 1)]
    
    a = 12       
    for i in range(a, a+1):        
    #for i in range(0, cluster_num + 1):         
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i]                 
            for code in each_cluster['stock_code']: 
                data = (pd.read_csv(str(code) + '.csv', header=None)
                        .iloc[:, 6].as_matrix()[::-1])                
                for j in range(start, term): 
                    d[i].append((pred_cluster(model, j, code)[2]) - (pred([data], j)[2]))
                                                                   
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                                               
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data) 
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    ax1.grid()
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability")
    plt.show()
    
#plot_origin_cluster(1, 30, 2) 


def plot_each_cluster(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    
    a = 12       
    for i in range(a, a+1): 
        d = []
        m = []       
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i]                 
            for code in each_cluster['stock_code']: 
                data = (pd.read_csv(str(code) + '.csv', header=None)
                        .iloc[:, 6].as_matrix()[::-1])                
                for j in range(start, term):  
                    d.append(pred_cluster(model, j, code)[2]) 
                    m.append(pred([data], j)[2])          
                           
        plot_data = [m, d]
                                               
        fig = plt.figure()    
        ax1 = fig.add_subplot(111)    
        ax1.boxplot(plot_data) 
        ax1.grid()
        plt.title('cluster' + str(i))
    plt.show()
    
#plot_each_cluster(1, 30, 2)    


def plot_csv_model():
    #term = 19
    #for term in range(term, term+1):
    for term in range(1, 31):
        d = [[] for i in range(20)] 
    
        for i in range(0,20):
            if i == 1 or i == 4 or i == 5 or i == 8 or i == 16: continue
            d[i] = pd.read_csv('model-2-4-' + str(i) + '.csv').iloc[:, term-1]
        
        plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                    d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                    d[17], d[18], d[19]]
        
        fig = plt.figure()    
        ax1 = fig.add_subplot(111)    
        ax1.boxplot(plot_data)
        ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
        ax1.grid()
        plt.legend(loc='upper right')
        plt.xlabel("cluster") 
        plt.ylabel("predicted probability")
        plt.show()
        
#plot_csv_model()  

       
def plot_csv_model_all():    
    d = [[] for c in range(20)]     
    for term in range(1, 31):    
        for cluster in range(20):
            if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
            d[cluster].append(pd.read_csv('model-2-4-' + str(cluster) + '.csv').iloc[:, term-1].as_matrix())
        
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                 
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data)
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
    ax1.grid()
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability")
    plt.show()
    
#plot_csv_model_all()   

     
 
def csv_chance_plot(start, term, model):
    cluster_data = pd.read_csv('nikkei_cluster_data.csv')
    cluster_num = int(cluster_data.iloc[[len(cluster_data)-1], [1]].as_matrix()) 
    n = map(int, np.linspace(start, term-1, term-1))
               
    for i in range(2, 3):        
    #for i in range(0, cluster_num + 1):            
        if len(cluster_data[cluster_data['cluster_number'] == i]) != 1: 
            each_cluster = cluster_data[cluster_data['cluster_number'] == i] 
            diff = [[] for c in range((term-1))] 
            
            for code in each_cluster['stock_code']: 
                row = pd.DataFrame() 
                            
                for j in range(start, term):                     
                    diff[j-1].append((pred_cluster(model, j, code)[2]) - 
                                     (pred_cluster(model, j, code)[3]))
                    row = pd.concat([row, pd.DataFrame(diff[j-1])], axis=1)  
                              
            result = pd.DataFrame(row.as_matrix(), columns=n)                   
            result.to_csv('chance-model' + str(model) + '-' + str(i) + '.csv', index=None) 
        
#csv_chance_plot(1, 31, 2) 


def plot_csv(model):
    #モデル4-精度高い5,13 低い24  モデル2-精度高い19,13 低い27,15,11
    term = 19
    #for term in range(term, term+1):
    for term in range(1, 31):
        d = [[] for i in range(20)] 
    
        for i in range(20):
            if i == 1 or i == 4 or i == 5 or i == 8 or i == 16: continue
            d[i] = pd.read_csv('model' + str(model) + '-' + str(i) + '.csv').iloc[:, term-1]
        
        plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                    d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                    d[17], d[18], d[19]]
        
        fig = plt.figure()    
        ax1 = fig.add_subplot(111)    
        ax1.boxplot(plot_data)
        ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
        ax1.grid()
        plt.legend(loc='upper right')
        plt.xlabel("cluster") 
        plt.ylabel("predicted probability")
        plt.show()
        
#plot_csv(2)  

def plot_csv_all(model):    
    d = [[] for c in range(20)]     
    for term in range(1, 31):    
        for cluster in range(20):
            if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
            d[cluster].append(pd.read_csv('model' + str(model) + '-' 
                              + str(cluster) + '.csv').iloc[:, term-1].as_matrix())
        
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                 
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data)
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
    ax1.grid()
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability")
    plt.show()
    
#plot_csv_all(2)  

            
            
def plot_csv_all(model):    
    d = [[] for o in range(20)]     
    for term in range(1, 31):    
        for cluster in range(20):
            if (cluster == 1 or cluster == 4 or cluster == 5 or 
                cluster == 8 or cluster == 16): continue
            a = pd.read_csv('model' + str(model) + '_' + str(cluster) + '.csv').iloc[:, term-1]
            b = pd.read_csv('model' + str(model) + '_chance_' + str(cluster) + '.csv').iloc[:, term-1]
            c = pd.read_csv('origin_' + str(cluster) + '.csv').iloc[:, term-1]
            d[cluster].append(((a - c) * 100))
            
    plot_data = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], 
                 d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], 
                 d[17], d[18], d[19]]
                 
    fig = plt.figure()    
    ax1 = fig.add_subplot(111)    
    ax1.boxplot(plot_data)
    ax1.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) 
    ax1.grid()
    plt.legend(loc='upper right')
    plt.xlabel("cluster") 
    plt.ylabel("predicted probability(%)")
    plt.show()
    
#plot_csv_all(2) 

  
"""  









