# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:13:59 2018

@author: BLAZIN
"""

import os

os.chdir("E:\\Python\\Resume Projects\\CBCA\\Stage 1 text Classificaiton")

import json
import pandas as pd
import numpy as np

#read the JSON file
chrg = pd.read_json("E:\\Python\\Resume Projects\\CBCA\\Stage 1 text Classificaiton\\Text Classification Data\\Charges.json", orient='records')

#Data preprocessing to convert string to float
from sklearn import preprocessing
def convert(data):
    num = preprocessing.LabelEncoder()
    data = num.fit_transform(data)
    return(data)

#Split to X and Y category
Y = convert(chrg.Category)
X = convert(chrg.Text)

#Split to train and test data
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

#load the Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB

#Gaussian NB

#Bernoulli NB
BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_Train, Y_Train)
print(BernNB)

Y_Exp = Y_Test
Y_Pred = BernNB.predict(Y_Train)
print accuracy_score(Y_Exp, Y_Pred)

#Multinomial NB
MultiNB = MultinomialNB()
MultiNB.fit(X_Train, Y_Train)
print(MultiNB)

Y_Exp = Y_Test
Y_Pred = MultiNB.predict(Y_Train)
print accuracy_score(Y_Exp, Y_Pred)

#Gausian NB
GausNB = GaussianNB()
GausNB.fit(X_Train, Y_Train)
print(GausNB)

Y_Exp = Y_Test
Y_Pred = GausNB.predict(Y_Train)
print accuracy_score(Y_Exp, Y_Pred)

#To find the accurace score of the model
from sklearn import metrics
print(metrics.classification_report())
print(metrics.confusion_matrix())
