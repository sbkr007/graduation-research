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
import matplotlib.dates as mdates

data = pd.read_csv('9984.csv', header=None)[::-1]
date = pd.to_datetime(data.iloc[:, 0]).as_matrix()
close = data.iloc[:, 6].as_matrix()

ax = plt.subplot()
ax.plot(date, close)
xfmt = mdates.DateFormatter("%m/%d")
xloc = mdates.MonthLocator()
ax.xaxis.set_major_locator(xloc)
ax.xaxis.set_major_formatter(xfmt)
plt.show()







