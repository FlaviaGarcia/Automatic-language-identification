# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:51:17 2020

@author: FlaviaGV, MatteoDM, CarlesBR, TheodorosPP
"""


import sys
sys.path.append("..")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np 


class LSTMSpeech:  
    
    def __init__(self, n_memory_cells, n_output_nodes, 
                 n_frames, n_features, dropout_rate_input=0.0, 
                 dropout_rate_hidden=0.0, optimizer_method="adam"):
        """
        

        Parameters
        ----------
        n_memory_cells : int 
            
        n_output_nodes : int
            
        n_frames : int
            Number of frames per utterance.
            
        n_features : int
            
        dropout_rate_input : float, optional
            Probability of neurons dropped of the input LSTM. The default is 0.0.
            
        dropout_rate_hidden : float, optional
            Probability of neurons dropped on the hidden layers. The default is 0.0.
            
        optimizer_method : string, optional
            Specify if you want to use adam optimizer or sgd. The default is "adam".

        """
        self.n_frames = n_frames
        self.n_features = n_features 
        self.n_memory_cells = n_memory_cells
        self.n_output_nodes = n_output_nodes
        self.dropout_rate_input = dropout_rate_input
        self.dropout_rate_hidden = dropout_rate_hidden
        self.output_activation_func = "softmax"
        self.optimizer_method = optimizer_method
        self.loss = 'categorical_crossentropy'

        self.create_NN()
        

    def create_NN(self):     
            
        self.model = Sequential()
        self.model.add(LSTM(self.n_memory_cells,
                            input_shape=(self.n_frames, self.n_features),
                            kernel_initializer='uniform',
                            unit_forget_bias='one', activation='tanh', 
                            recurrent_activation='sigmoid', 
                            return_sequences=True,
                            dropout=self.dropout_rate_input, 
                            recurrent_dropout=self.dropout_rate_hidden))
        
        self.model.add(Dropout(self.dropout_rate_hidden))
        
        self.model.add(Dense(self.n_output_nodes, activation="softmax"))

        if self.optimizer_method == "sgd":
            optimizer_method = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
        else: 
            optimizer_method = Adam()
            
        self.model.compile(loss=self.loss, optimizer=optimizer_method, metrics=['accuracy'])

    
    def convert_data(self, X, y=np.array(None)): 
        """
        Convert the input data into the data accepted by LSTM that is of 3D: 
        (n_samples, n_time_steps, n_features)

        Parameters
        ----------
        X : numpy shape=(n_utterances*self.n_frames, self.n_features)
            Features matrix.
            
        y : numpy shape=(n_utterances_train*self.n_frames, self.n_output_nodes)
            Targets matrix. The default is np.array(None).

        Raises
        ------
        ValueError
            If the number of frames doesnt match the number of rows of X.

        Returns
        -------
        X_conv : numpy shape=(n_utterances, self.n_frames, self.n_features)
        y_conv : numpy shape=(n_utterances, self.n_frames, self.n_output_nodes)

        """
        if X.shape[0] % self.n_frames != 0:
            raise ValueError("The number of frames per utterance doesnt match the input dimension ")
        
        n_utterances = X.shape[0]//self.n_frames
        
        X_conv = X.reshape(n_utterances, self.n_frames, self.n_features)
        
        if y.shape != ():
            y_conv = y.reshape(n_utterances, self.n_frames, -1)
        else:
            y_conv=None
            
        return X_conv, y_conv

        

    def train(self, features_train, targets_train, features_val, targets_val, 
              batch_size, n_epochs, verbose=1):
        """
        
        Parameters
        ----------
        features_train : numpy shape=(n_utterances_train*self.n_frames, self.n_features)

        targets_train : numpy shape=(n_utterances_train*self.n_frames, self.n_output_nodes)
            
        features_val : numpy shape=(n_utterances_val*self.n_frames, self.n_input_nodes)
        
        targets_val : numpy shape=(n_utterances_val*self.n_frames, self.n_output_nodes)
        
        batch_size : int      
        
        n_epochs : int       
        
        verbose : int, the default is 1.

        Returns
        -------
        model_info : TYPE

        """
        
        features_train, targets_train = self.convert_data(features_train, targets_train)
        features_val, targets_val = self.convert_data(features_val, targets_val)

        model_info = self.model.fit(features_train, targets_train, batch_size=batch_size,  
                                    validation_data=(features_val, targets_val), 
                                    verbose=verbose, epochs=n_epochs)
    
        return model_info
    
    
    def predict_proba(self, features): 
        """
        Parameters
        ----------
        features : numpy shape=(n_utterances*self.n_frames, self.n_features)

        
        Returns
        -------
        proba_utterances: numpy shape=(n_utterances, self.n_output_nodes)
            Average of the logarithm of the sotmax values of frames per utterance.
            
        """ 
        features, _ = self.convert_data(features)
        
        sotmax_scores = self.model.predict_proba(features)
        
        proba_utterances = np.mean(np.log(sotmax_scores), axis=1) 
            
        return proba_utterances
    
    
    def predict_classes(self, features):
        """
        Get network class prediction for each utterance.
        
        Parameters
        ----------
        features : numpy shape=(n_utterances*self.n_frames, self.n_features)

        Returns
        -------
        utterances_classes: numpy shape=(n_utterances,).
        """
        proba_utterances = self.predict_proba(features)
        utterances_classes = np.argmax(proba_utterances, axis=1)
        return utterances_classes

