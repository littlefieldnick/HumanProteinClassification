#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:23:15 2018

@author: nicklittlefield
"""

from keras.callbacks import *
import numpy as np


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=1000):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

def exp_decay_schedule(learning_rate, decay_rate=0.96, decay_steps=1000):
    '''
    Wrapper function to create a LearningRateScheduler with exponential decay schedule
    '''
    def schedule(epoch):
        step = np.min(epoch, decay_steps)
        return learning_rate * decay_rate**(step/decay_steps)
    
    return LearningRateScheduler(schedule)

def poly_decay_schedule(learning_rate, decay_steps=1000, end_learning_rate = 0.001, power=1.0):
    '''
    Wrapper function to create a LearningRateScheduler with exponential decay schedule
    '''
    def schedule(epoch):
        step = np.min(epoch, decay_steps)
        return (learning_rate - end_learning_rate) * (1 - step/decay_steps) ** power + end_learning_rate
    
    return LearningRateScheduler(schedule)


def linear_cosine_decay(learning_rate, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001):
    '''
    Wrapper function to create a LearningRateScheduler with linear cosine decay schedule
    '''
    
    def schedule(epoch):
        step = np.min(epoch, decay_steps)
        linear_decay = (decay_steps - step)/decay_steps
        cosine_decay = 0.5 * (1 + np.cos(np.pi * 2 * num_periods * step/decay_steps))
        decayed = (alpha + linear_decay) * cosine_decay + beta
        decayed_learning_rate = learning_rate * decayed
        return decayed_learning_rate
    
    return LearningRateScheduler(schedule)

def inverse_time_decay(learning_rate, decay_steps, decay_rate):
    '''
    Wrapper function to create a LearningRateScheduler with inverse time decay schedule
    '''
    
    def schedule(epoch):
         return learning_rate / (1 + decay_rate * epoch /decay_steps)
     
    return LearningRateScheduler(schedule)
        