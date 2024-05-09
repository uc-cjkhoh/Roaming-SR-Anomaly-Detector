# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:46:19 2024 
@author: cj_khoh
"""
  
import pandas as pd  

from collections import Counter 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
  
class ML_Model:  
    def dbscan(self, data, target_column, lag=None, pca=False):
        """Train DBSCAN model

        Args:
            data (DataFrame): actual dataset to train model
            target_column (String): The target column to classify
            lag (Integer, optional): Drag N rows to the next row
            pca (Boolean, optional): Use PCA to reduce dimension if true  

        Returns:
            list: label
            integer: most_common_label 
        """
        
        std = data[target_column].std()
        
        model = DBSCAN(eps=std)
        label = None

        if lag == None:
            label = model.fit_predict(data)
        else:
            data = pd.DataFrame(data)

            lag_columns = pd.concat([data[target_column].diff(i).rename(f'lag{i+1}') for i in range(1, lag+1)], axis=1)
            data = pd.concat([data, lag_columns], axis=1)
                
            data.dropna(inplace=True)
            data = data.to_numpy()
            
            if lag >= 3 and pca:
                pca = PCA(n_components=3)
                pca.fit_transform(data)
 
            label = model.fit_predict(data)

        most_common_label = Counter(label).most_common(1)[0][0]

        return label, most_common_label
 
 