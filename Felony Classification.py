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
import matplotlib.pyplot as plt
import seaborn as sns

chrg = pd.read_json("E:\\Python\\Resume Projects\\CBCA\\Stage 1 text Classificaiton\\Text Classification Data\\Charges.json", orient='records')
chrg.Category

train = chrg(subset='train', categories = chrg.Category)

df = pd.DataFrame(chrg)


#-----------------------------------------------------------------------------------------------------------------------
from sklearn import datasets

news = datasets.fetch_20newsgroups()

iri = datasets.load_iris()
print(iri)