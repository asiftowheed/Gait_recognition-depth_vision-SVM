# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:12:03 2019

@author: Asif Towheed
"""
import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct

from skimage import io
import matplotlib.pyplot as plt
from sklearn import svm
import pyrealsense2 as rs
import cv2
from PIL import Image, ImageEnhance
from PIL import ImageDraw
import re
import math
import pickle
import time
from os.path import isfile, join
import video

from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  

correct= 0;
total = 0;
y_pred = np.empty((1))
y_test = np.empty((1))

infile = open(r'C:/Users/Asif Towheed/Documents/DEVEL/Gait_Recognition/AD/trained-model.pkl','rb')
clf = pickle.load(infile)

for i in os.listdir(r'C:\Users\Asif Towheed\Documents\DEVEL\Gait_Recognition\AD'):                                          # open each walk for recorded person
    walks = []                                              # array to store how many times they have walked
    label_path = r'C:\Users\Asif Towheed\Documents\DEVEL\Gait_Recognition\AD'.replace('\\','/') + '/' + i                          # path to that person        
        
    if i == 'trained-model.pkl' or i in ['ABDALLAH','HIND','HANIA','SOMAR','REZA']:
        continue
    for walk in os.listdir(label_path):
        if os.path.exists('C:/Users/Asif Towheed/Desktop/Trimmed/K_A/' + i + '/' + walk):
            continue
        print (i + "'s " + walk + ' now running!')
#        infile = open(label_path + '/' + walk + '/fvec_dof.pkl','rb')
#        imtest = pickle.load(infile) 
        imtest = np.array(Image.open(label_path + '/' + walk + '\AD.jpeg').convert("L"))

    
        dcttest = dct( dct( imtest, axis=0), axis=1)
    
        featurevec2 = []
        for m in range(10):
            for n in range(10):
                featurevec2.append(dcttest[m][n])
    
    
        fvectest = np.array(featurevec2)
    
        fvectest = fvectest.reshape(1,-1)
    
    #    y_pred = clf.predict_proba(fvectest)
    #    print(y_pred)
        total += 1
        if(total == 1):
          y_pred = clf.predict(fvectest)
          y_test = i
        y_pred = np.append(y_pred, clf.predict(fvectest))
        y_test = np.append(y_test, i)
#        if y_pred == i:
#            correct += 1
            
#print ('CoRrEcT = ' + str(correct) + '\n')
#print ('tOtAl = ' + str(total) + '\n')
#print ('accuracy = ' + str(float(correct)/total) + '\n')


cm = confusion_matrix(y_test,y_pred)
print(cm)  
cr = classification_report(y_test,y_pred)
print(cr)

            