#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
path = '/Users/anthonybaptista/Downloads/fMNIST_DNN_training/fmnist/'
os.chdir(path)

test = pd.read_csv("fashion-mnist_test.csv")
train = pd.read_csv("fashion-mnist_train.csv")

# extract label 5 and 9

def extract(data, label, name_data, save = True):
    data_extracted = data[data['label'] == label]
    if save == True:
        data_extracted.to_csv(name_data + str(label) + ".csv", index = False)
    
extract(test, 5, 'test')
extract(test, 9, 'test')
extract(train, 5, 'train')
extract(train, 9, 'train')

