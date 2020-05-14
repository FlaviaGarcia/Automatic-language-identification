# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:00:55 2020

@author: FlaviaGV, MatteoDM, CarlesBR, TheodorosPP
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from models import utils

class DNN:  
    
    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes, 
                 batch_normalization, dropout_rate=0.0):
        """

        Parameters
        ----------
        n_input_nodes : int
        
        n_hidden_nodes : list of ints
            number of nodes per hidden layer.
        
        n_output_nodes : int
        
        batch_normalization : boolean
                
        dropout_rate : float, the default is 0.0. This means that no dropout is applied. 

        Raises
        ------
        ValueError
            it is raised if n_hidden_nodes is not a list.

        """
        self.n_input_nodes = n_input_nodes
        
        if type(n_hidden_nodes) != list:
            raise ValueError("n_hidden_nodes has to be a list specifying the \
                            number of nodes per layer")
        
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = len(self.n_hidden_nodes)

        self.n_output_nodes = n_output_nodes
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

        self.activation_func = "relu"
        self.output_activation_func = "softmax"
        self.optimizer_method = "adam"
        self.loss = 'categorical_crossentropy'

        self.create_NN()
        

    def create_NN(self):     
        
        self.model = Sequential()
        
        self.model.add(Dense(self.n_hidden_nodes[0], input_dim=self.n_input_nodes, 
                        activation=self.activation_func))

        self.model.add(Dropout(self.dropout_rate))
        
        for idx_hidden in range(1, self.n_hidden_layers):
            if self.batch_normalization:
                self.model.add(BatchNormalization())
            
            self.model.add(Dense(self.n_hidden_nodes[idx_hidden], 
                                 activation=self.activation_func))
            
            self.model.add(Dropout(self.dropout_rate))
            
        if self.batch_normalization:
            self.model.add(BatchNormalization())
        
        self.model.add(Dense(self.n_output_nodes, 
                             activation=self.output_activation_func))
    
        self.model.compile(loss = self.loss, 
                           optimizer = self.optimizer_method, 
                           metrics = ['accuracy'])  ## not sure about this   
        

    def train(self, features_train, targets_train, features_val, targets_val, 
              batch_size, n_epochs, verbose=1):
        """
        
        Parameters
        ----------
        features_train : numpy shape=(n_samples_train, self.n_input_nodes)

        targets_train : numpy shape=(n_samples_train, self.n_output_nodes)
            
        features_val : numpy shape=(n_samples_val, self.n_input_nodes)
        
        targets_val : numpy shape=(n_samples_val, self.n_output_nodes)
        
        batch_size : int      
        
        n_epochs : int       
        
        verbose : int, the default is 1.

        Returns
        -------
        model_info : TYPE

        """
        model_info = self.model.fit(features_train, targets_train, batch_size=batch_size,  
                               validation_data=(features_val, targets_val), 
                               verbose=verbose, epochs=n_epochs)
    
        return model_info
    


    def predict_proba(self, features, n_frames_utterance): 
        """

        Parameters
        ----------
        features : numpy shape=(n_samples, self.n_input_nodes)

        n_frames_utterance : int
            Number of frames per utterance.

        Raises
        ------
        ValueError
            If there are more frames than expected in features.

        Returns
        -------
        proba_utterances: numpy shape=(n_utterances, self.n_output_nodes)
            Average of the logarithm of the sotmax values of frames per utterance.
            
        """
        n_samples = features.shape[0]
        
        if n_samples%n_frames_utterance != 0:
            raise ValueError("More frames that expected in the features matrix")
        
        n_utterances = n_samples//n_frames_utterance
        
        proba_utterances = np.zeros((n_utterances, self.n_output_nodes)) # probabilities of each utterance to belong to each class
        
        
        sotmax_scores = self.model.predict_proba(features)
        
        # Do average logarithm of the sotmax scores of each utterance
        for idx_utterance in range(n_utterances):
            idx_start_frames = idx_utterance * n_frames_utterance
            idx_end_frames = (idx_utterance+1) * n_frames_utterance
            softmax_scores_utterance = sotmax_scores[idx_start_frames:idx_end_frames]
            proba_utterances[idx_utterance] = utils.avg_log_scores(softmax_scores_utterance)
            
        return proba_utterances
    
    
    def predict_classes(self, features, n_frames_utterance):
        """
        Get network class prediction for each utterance.
        
        Parameters
        ----------
        features : numpy shape=(n_samples, self.n_input_nodes) 

        n_frames_utterance : int
            Number of frames per utterance.

        Returns
        -------
        utterances_classes: numpy shape=(n_utterances,).
        """
        proba_utterances = self.predict_proba(features, n_frames_utterance)
        utterances_classes = np.argmax(proba_utterances, axis=1)
        return utterances_classes

    