# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024 
@author: cj_khoh
"""

import util 
import pandas as pd  
 
from datetime import datetime
from impala.dbapi import connect
from sklearn.preprocessing import MinMaxScaler
  
# configure and queries
IMPALA_HOST = '10.168.49.12'
IMPALA_PORT = 21050

class Dataset:
    def __init__(self, window_size, query=None, qh_grouping=None, save_data=False):
        """_summary_

        Args:
            window_size (Integer): Select N data point as one subset
            query (String, optional): Complete query to execute
            qh_grouping (Integer, optional): Moving Average Range
            save_data (Boolean, optional): Save data to current filepath
        """
        
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
        
        # fill null value and scaling
        scaler = MinMaxScaler()
        for column in self.data.columns:
            self.data[column].fillna(self.data[column].median(), inplace=True)
            self.data[column] = scaler.fit_transform(self.data[column].values.reshape(-1, 1))

        if qh_grouping != None:
            self.data['success_rate'] = self.data['success_rate'].rolling(qh_grouping).mean()
            self.data.dropna(inplace=True)
        
        if save_data:
            self.data.to_csv(r'from_sql_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
                 
     
    # return dataset
    def get_data(self):
        return self.data
    