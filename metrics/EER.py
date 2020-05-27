# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:50:38 2020

@author: FlaviaGV
"""

import numpy as np
from sklearn.metrics import roc_curve

def val2onehot(val_array, classes):
    labels = np.zeros((len(val_array), classes))
    for ind,lbl in enumerate(val_array):
        labels[ind,lbl] = 1
    return labels

def EER(true_targets_onehot, predictions):
    '''
    Imputs :
        true_targets_onehot: one hot encoding of true values, shape (n_samples x n_classes)
        predictions: one hot encoding of models prediction, shape (n_samples x n_classes)
    Output:
        per class EER score, np vector of length (n_clases,)

    !!!!Be carefull!!! check that the target_to_class dictionary is the same for the training and the test, 
    if not you are returning incorect classes
    Use predictions = model.predict(data_gen_val, verbose=1) to get the predictions,
    and true_targets_onehot = val2onehot(data_gen_val.getTargets(), 8) to get the true targets

    '''
    scores = []
    for i in range(true_targets_onehot.shape[1]):
        fpr, tpr, threshold = roc_curve(true_targets_onehot[:,i], predictions[:,i])
        fnr = 1 - tpr
        ballanced_thres_pos = np.nanargmin(np.absolute((fnr - fpr)))
        scores.append(fpr[ballanced_thres_pos])
    return np.array(scores)