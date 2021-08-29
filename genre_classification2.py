#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 03:49:16 2021

@author: adee
"""

#importing modules
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import os
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import keras
from keras import models
from keras import layers


# generating a dataset
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
#print(header)
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

#genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
#removed jazz coz error in loading the folder and one of the files had weird format.
genres = 'blues classical country disco hiphop metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'/Users/adee/Desktop/SDEProj/Data/genres_original/{g}'):
        songname = f'/Users/adee/Desktop/SDEProj/Data/genres_original/{g}/{filename}'
        #print(songname)
        y, sr = librosa.load(songname, mono=True, duration=30)
        #print(len(y),y,sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        #print(to_append)
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

# reading dataset from csv

data = pd.read_csv('data.csv')
data.head()

# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)
data.head()
#print(data.shape[1])

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
#print(y)

# normalizing
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
#print(X)

# spliting of dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(y_train),len(y_test))


# creating a model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))
#model.output_shape
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#used adam coz best adaptive optimizer, and sparse_categorical_crossentropy because the labels are numeric and have more than 2 classes
history = model.fit(X_train,y_train,epochs=20,batch_size=72)  #batch size must divide len(X_train)
                    
# calculate accuracy
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)

# predictions
predictions = model.predict(X_test)
np.argmax(predictions[0])
