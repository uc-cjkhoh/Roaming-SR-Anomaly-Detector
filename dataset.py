# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024

@author: cj_khoh
"""

import sys
sys.path.append(r'C:\Users\cj_khoh\Documents\UnifiedComms\Scripts\Python\time series - anomalies detection')
 
import util 
import pandas as pd
import numpy as np  
pd.set_option("display.max_rows", None)
 
from datetime import datetime
from impala.dbapi import connect     
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
  
# configure and queries
IMPALA_HOST = '10.168.49.12'
IMPALA_PORT = 21050

class Dataset:
    def __init__(self, window_size, query=None, qh_grouping=None, save_data=True):
        # initialize connection
        conn = connect(host=IMPALA_HOST, port=IMPALA_PORT)
        cursor = conn.cursor()    
        cursor.execute(query)
        self.data = pd.DataFrame(cursor.fetchall(), columns=pd.DataFrame(cursor.description).iloc[:, 0].values).sort_values('dt').set_index('dt')
        self.data.index = pd.to_datetime(self.data.index)
        
        # filling missing data
        self.data = util.reindex(self.data, freq="H")

        # preprocessing
        self.data = util.filter_unwanted_value(self.data)
        self.data['weekly_diff'] = self.data['success_rate'].diff(window_size)
        self.data['weekly_diff'] = self.data['weekly_diff'].fillna(self.data['weekly_diff'])
        self.data['daily_diff'] = self.data['success_rate'].diff(1) 

        poly = np.poly1d(np.polyfit(np.arange(0, len(self.data), 1), self.data['success_rate'].to_numpy(), 4))
        pred = poly(np.arange(0, len(self.data), 1))

        self.data['euclidean'] = (self.data['success_rate'].values - pred) ** 2 
        
        # fill null value and scaling
        scaler = MinMaxScaler()
        for column in self.data.columns:
            self.data[column].fillna(self.data[column].median(), inplace=True)
            self.data[column] = scaler.fit_transform(self.data[column].values.reshape(-1, 1))

        # generate day, hour column
        # self.data['dayofweek'] = self.data.index.to_series().dt.dayofweek
        # self.data['hour'] = self.data.index.to_series().dt.hour 
       
        if qh_grouping != None:
            self.data['success_rate'] = self.data['success_rate'].rolling(qh_grouping).mean()
            self.data.dropna(inplace=True)
        
        if save_data:
            self.data.to_csv(r'C:\Users\cj_khoh\Documents\UnifiedComms\Scripts\Python\time series - anomalies detection\input\from_sql_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
                
        # convert to dataframe format and preprocessing
        self.window_size = window_size 
        self.x, self.y = [], []

        self.custom_xy(to_diff=False, ML='None')


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        return self.data.iloc[idx, :]
    
    
    # function to preprocess data / create x, y for training loop
    def custom_xy(self, to_diff=False, ML=None):
        temp = self.data

        if ML != None:
            lr = LinearRegression()
            lr.fit(np.arange(0, len(temp), 1).reshape(-1, 1), self.data.to_numpy())

        if to_diff:
            temp = temp.diff()[1:]
 
        for i in range(len(temp) - self.window_size):  
            self.x.append(np.array(temp.iloc[i:i+self.window_size]))

            if ML == 'linear':
                target = np.array(lr.intercept_ + (lr.coef_ * (self.window_size + i)))
                self.y.append(target)
            else:
                self.y.append(np.array(temp.iloc[i+self.window_size]))  


    def get_data(self):
        return self.data
     