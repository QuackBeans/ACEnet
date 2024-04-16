

# This is a set of functions for preprocessing the NinaPro datasets, keeping the other files clean.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import os
from keras.models import Sequential, Model, load_model
import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras as K
import matplotlib.pyplot as plt


# Organise the data in to a dataframe
def get_data(path,file):
    mat = loadmat(os.path.join(path,file))
    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus'] 
    data['repetition'] = mat['repetition']
    
    return data

# Normalise the data by specified repetitions in its dataframe using the scikit standardscaler
def normalise(data, train_reps):
    x = [np.where(data.values[:,13] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis = -1))
    train_data = data.iloc[indices, :]
    train_data = data.reset_index(drop=True)
    
    scaler = StandardScaler(with_mean=True,
                                with_std=True,
                                copy=False).fit(train_data.iloc[:, :12])
    
    scaled = scaler.transform(data.iloc[:,:12])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus']
    normalised['repetition'] = data['repetition']
    return normalised


# Window the time series data in to samples by specified repetitions
def windowing(data, reps, gestures, win_len, win_stride):
    if reps:
        x = [np.where(data.values[:,13] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    if gestures:
        x = [np.where(data.values[:,12] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    idx=  [i for i in range(win_len, len(data), win_stride)]
    
    X = np.zeros([len(idx), win_len, len(data.columns)-2])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])
    
    for i,end in enumerate(idx):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:12].values
        y[i] = data.iloc[end, 12]
        reps[i] = data.iloc[end, 13]
        
    return X, y, reps


# One-hot encode the labels
def get_categorical(y):
    return pd.get_dummies(pd.Series(y)).values
