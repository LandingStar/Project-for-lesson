from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(24, input_dim=input_dim, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model