# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:00:55 2020

@author: FlaviaGV
"""

from keras.models import Sequential
from keras.layers.core import Dense, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import numpy as np
import utils

class LSTM:  
    
    def __init__(self, n_memory_cells):
        self.n_memory_cells = n_memory_cells
        self.output_activation_func = "softmax"
        self.optimizer_method = "adam"
    
    def create_NN(self):   
        pass
    
    
    def train(self, features_train, targets_train, features_val, targets_val, 
              batch_size, n_epochs, verbose=1):
        pass 
    
"""
n_memory_cells = 512
"""    