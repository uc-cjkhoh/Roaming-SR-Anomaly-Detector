# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:46:19 2024

@author: cj_khoh
"""
  
import pandas as pd 
import numpy as np 

from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class Labelling_Model:   
    def svm_classifier(self, x):
        svm = OneClassSVM(nu=0.1, degree=4)
        label = svm.fit_predict(x)

        return label
    
    def isolationForest(self, x):
        lf = IsolationForest()
        label = lf.fit_predict(x)

        return label
    
    def svm_with_if(self, x):
        svm_label = Labelling_Model().svm_classifier(x)
        lf_label = Labelling_Model().isolationForest(x)

        return np.where((svm_label == -1) & (lf_label == -1), -1, 1)

    def percent_diff(self, x, y):
        # target percentage
        percent = 100 - (x[y] / x[y].shift(24 * 7)) * 100
        q1 = np.quantile(percent.dropna().values, 0.25)
        q3 = np.quantile(percent.dropna().values, 0.75)

        IQR = 1.5 * (q3 - q1) 

        percent_diff = 100 - (x[y] / x[y].shift(24 * 7)) * 100
        return np.where(abs(percent_diff) > q3 + IQR, -1, 1) 
        
  
class ML_Model:  
    def dbscan(self, data, lag=None, pca=False):
        std = data['success_rate'].std()
        
        model = DBSCAN(eps=std)
        label = None

        if lag == None:
            label = model.fit_predict(data)
        else:
            data = pd.DataFrame(data)

            lag_columns = pd.concat([data['success_rate'].diff(i).rename(f'lag{i+1}') for i in range(1, lag+1)], axis=1)
            data = pd.concat([data, lag_columns], axis=1)
                
            data.dropna(inplace=True)
            data = data.to_numpy()
            
            if lag >= 3 and pca:
                pca = PCA(n_components=3)
                pca.fit_transform(data)
 
            label = model.fit_predict(data)

        most_common_label = Counter(label).most_common(1)[0][0]

        return label, most_common_label
 
 