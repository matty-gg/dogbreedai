#REQUIRED IMPORTS
import matplotlib.pyplot as plt
import csv
import cv2
import xml.etree.ElementTree as ET


import tensorflow as tf
import numpy as np
import pandas as pd

import time
from datetime import timedelta

import math
import os

import scipy.misc
#from scipy.stats import itemfreq
from random import sample
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Image manipulation
import PIL.Image
from IPython.display import display

# Open a Zip File
from zipfile import ZipFile
from io import BytesIO

# Check data
df_labels = pd.read_csv("labels.csv")
#print("Total number of unique Dog Breeds :",len(df_labels.breed.unique()))
#print(len(df_labels.breed))

# reduce data size
temp_set = df_labels['breed'].unique()[:60]

new_set = []
for index, row in df_labels.iterrows():
    if row['breed'] in temp_set:
        new_set.append(row)

df_train = pd.DataFrame(new_set)

df_train.to_csv('new_set.csv', index = False)






