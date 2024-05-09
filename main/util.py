# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:38:15 2024

@author: cj_khoh
"""
   
from sklearn.decomposition import PCA  
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np 
import model
 

def reindex(data, freq=None):
    if freq == None:
        raise ValueError('Missing Frequency Argument')

    ori_column = data.columns

    idx = pd.date_range(
        start=data.index[0], 
        end=data.index[-1],
        freq=freq
    )

    return data.reindex(idx, fill_value=None, columns=['dt'].append(ori_column))


def filter_unwanted_value(data, threshold=None):
    if threshold:
        data[data['success_rate'] < threshold] = None
    
    data['success_rate'] = data['success_rate'].interpolate(option='spline')

    return data


def dimension_reduction(data):
    pca = PCA(n_components=1)
    result = pca.fit_transform(data)

    return result


def plot_euclidean_distance(data, y):
    poly = np.poly1d(np.polyfit(np.arange(0, len(data), 1), data[y].to_numpy(), 3))
    pred = poly(np.arange(0, len(data), 1))  
    
    x = data.index
    y = abs((data['success_rate'].values - pred) ** 2) 
    
    x = np.repeat(x,3)
    y = np.repeat(y,3)
    y[::3] = y[2::3] = 0
 
    fig = px.line(x=x, y=y) 
    fig.show()

 
def plot_chart(df, x, y, label=None):
    """
    Plot labelling result on actual data

    Args:
        x (DataFrame): Preprocessed Dataset 
        y (String): Target Column
        label (narray): Unsupervised labelling result 
    """
    poly = np.poly1d(np.polyfit(np.arange(0, len(df), 1), df[y].to_numpy(), 4))

    fig = px.line(x=df.index, y=df[y]).update_layout(xaxis_title=x, yaxis_title=y) 
    fig.add_trace(go.Scatter(x=df.index, y=poly(np.arange(0, len(df), 1)), name='Polynomial Best Fit', mode='lines', marker_color='black'))

    if label is not None and label.any():  
        fig.add_trace(go.Scatter(x=df[label == -1].index, y=df[label == -1][y], mode='markers', marker_color='red', name='Outliers'))
     
    fig.update_traces(hovertemplate=None)
    fig.update_layout(hovermode='x unified')
    fig.show() 
    

def summary(label):
    """
    Print a summary of this anomaly detection
    1. Outlier Percentage 

    Args:
        label (List): Unsupervised labelling result 
    """
    print('Outlier Percentage: {}%\n'.format(round((len(label[label == -1]) / len(label)) * 100, 4)))
