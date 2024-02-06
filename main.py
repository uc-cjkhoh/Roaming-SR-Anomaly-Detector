# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:26:20 2024

@author: cj_khoh
""" 
import sys
sys.path.append(r'C:\Users\cj_khoh\Documents\UnifiedComms\Scripts\Python\time series - anomalies detection')

import util
import model
import pandas as pd
import dataset 
import numpy as np
import plotly.express as px

from collections import Counter

if __name__ == '__main__':   
   QUERY = ''
   # FILEPATH = r'C:\Users\cj_khoh\Documents\UnifiedComms\Scripts\Python\time series - anomalies detection\April_to_May.csv'

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
   
   _data = [data, temp]

   """
   for i, d in enumerate(_data):
      data_label, data_common_label = model.ML_Model().dbscan(d)
      util.plot_chart(x = d, y='success_rate', label = np.where(data_label == data_common_label, 1, -1))  
      util.plot_3d(d.copy())

      print('Dataset {}'.format(i+1))
      util.summary(np.where(data_label == data_common_label, 1, -1))
   """
      
   util.plot_euclidean_distance(temp, 'success_rate')