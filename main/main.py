# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024

@author: cj_khoh
"""  

import util
import model
import dataset 
import numpy as np 

if __name__ == '__main__':   
   QUERY = '' 

   # read sql query from .txt
   with open(r'succ_rate.txt') as file:
      QUERY = file.readlines()[0]

   # configuration
   window_size_per_day = 24
   data = dataset.Dataset(
      window_size=window_size_per_day * 7, 
      query=QUERY, 
      qh_grouping=None
   ).get_data() 
 
   temp = util.filter_unwanted_value(data.copy(), threshold=0.3)
   temp['success_rate'] = temp['success_rate'].interpolate(option='spline') 
   temp['weekly_diff'] = temp['weekly_diff'].fillna(temp['weekly_diff'].median())
   temp.fillna(0, inplace=True)
   
   _data = [data[['success_rate']], temp[['success_rate']]]

   # for collective anomalies 
   collective_labels = [None, None]   
   for i, d in enumerate(_data):
      data_label, data_common_label = model.ML_Model().dbscan(d, target_column='success_rate', pca=True)
      label = np.where(data_label == data_common_label, 1, -1)
      collective_labels[i] = label

      # util.plot_chart(df=d, x='Datetime', y='success_rate', label = label)  
   
   # for contextual anomalies
   contextual_labels = [None, None]   
   for i, d in enumerate(_data):
      poly = np.poly1d(np.polyfit(np.arange(0, len(d), 1), d['success_rate'].to_numpy(), 4))
      pred = poly(np.arange(0, len(d), 1))

      d = d.copy()
      d.loc[:, 'euclidean'] = (d['success_rate'].values - pred) ** 2
      label = np.where(d['euclidean'] < np.percentile(d['euclidean'], 95), 1, -1)
      contextual_labels[i] = label

      # util.plot_chart(df=d, x='Datetime', y='success_rate', label = label)

   # complete graph
   for i, d in enumerate(_data):
      util.plot_chart(d, x='Datetime', y='success_rate', label = np.where((collective_labels[i] == -1) | (contextual_labels[i] == -1), -1, 1))
