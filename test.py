import pandas as pd
import math
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import sklearn.model_selection as cv
import sklearn.datasets as datasets
import collections
from statsmodels.tsa.arima.model import ARIMA 
# import stats
from tensorflow import keras
from tensorflow.keras import layers, backend, utils, Input, models
import processing
import warnings
import matplotlib.pyplot as plt



input = keras.Input(shape=(5,13),name='input')
x = layers.LSTM(150)(input)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
output = layers.Dense(1, activation='linear', name='output')(x)
model = models.Model(inputs=[input], outputs=[output])
dot_img_file = 'LSTM_architecture.png'
utils.plot_model(model, to_file=dot_img_file, show_shapes=True)