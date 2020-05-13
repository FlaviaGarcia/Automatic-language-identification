# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:21:25 2020

@author: FlaviaGV, MatteoDM, CarlesBR, TheodorosPP
"""

import numpy as np 

def avg_log_scores(scores):
    """
    Parameters
    ----------
    scores : numpy shape=(n_frames, n_classes)
        softmax scores of the frames of one utterance 
        
    Returns
    -------
    numpy shape=(n_classes,)
        average of the logarithm of the scores for the utterance

    """
    return np.mean(np.log(scores), axis=0)


def generate_fake_data(n_utterances=3, n_frames_utterance=10, n_features=10, 
                       n_utt_other_class=1):
    """
    Generate fake features and targes matrices. The targets would be just 0 or 1. 

    Parameters
    ----------
    n_utterances : int

    n_frames_utterance : int
        
    n_features : int
    
    n_utt_other_class: int
        number of utterances of class 1

    Returns
    -------
    fake_features : numpy shape=(n_utterances * n_frames_utterance, n_features)

    fake_targets: numpy shape=(n_utterances * n_frames_utterance)
        All utterances would be class 0 except the n_utt_other_class last ones.
    """
    n_rows = n_utterances * n_frames_utterance
    fake_features = np.random.randn(n_rows, n_features)
    fake_targets = np.zeros(n_rows)
    fake_targets[-n_utt_other_class*n_frames_utterance:] = 1
    return fake_features, fake_targets