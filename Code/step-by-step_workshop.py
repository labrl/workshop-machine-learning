# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:06:28 2020

@author: Marc Lanovaz
"""

#Set working directory (change for your path)
import os 
os.chdir('PATH')

#Import AB data
import pandas as pd

x_ab = pd.read_csv('x_original_data.csv')
x_ab = x_ab.values

y_ab = pd.read_csv('y_original_data.csv', header = None)
y_ab = (y_ab.values).flatten()

#Import IWT data
iwt = pd.read_excel('TurgeonetalData.xlsx')
x_iwt = iwt.values[:,0:4]
y_iwt = iwt.values[:,4]

#Import VS data
import numpy as np
x_vs = np.load('x_all.npy')
y_vs = np.load('y_all.npy')

#Standardize values for iwt
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()
x_iwt[:,2:4] = standard_scaler.fit_transform(x_iwt[:,2:4])

#Stardardize values for vs
for i in range(x_vs.shape[1]):
    x_vs[:, i, :] = standard_scaler.fit_transform(x_vs[:, i, :]) 

#Outcome measures
from sklearn.metrics import accuracy_score, cohen_kappa_score

#Comparison measure for IWT
np.random.seed(48151)
y_random_iwt = []

for i in range(10000):
    y_random_values = np.random.choice(y_iwt, 26, replace = False)
    y_random_iwt.append(np.sum(y_iwt==y_random_values)/26)

print(np.mean(y_random_iwt))
print(np.sum(y_iwt)/26)

#Comparison measure for VS
y_random_vs = []

for i in range(1000):
    y_random_values = np.random.choice(y_vs, 99622, replace = False)
    y_random_vs.append(np.sum(y_vs==y_random_values)/99622)

print(np.mean(y_random_vs))
print(np.sum(y_vs==0)/99622)

#Cross-validation for AB study
from sklearn.model_selection import train_test_split
x_train_ab, x_test_ab, y_train_ab, y_test_ab =\
    train_test_split(x_ab, y_ab, test_size = 0.20, random_state = 48151)

#Import random forest function
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight = 'balanced', random_state = 48151)

#Train rf model for AB study
rf.fit(x_train_ab, y_train_ab)
    
#Get prediction for untrained data on rf model for AB study
predictions_ab = rf.predict(x_test_ab)

#Get rf accuracy and kappa for AB study
print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))

###Applying rf to IWT data

#Set leave-one out method
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

#Create empty list to get result from each of the 26 repetitions
rf_pred = []

#Run loop for leave-one out cross-validation
for train_index, test_index in loo.split(x_iwt): #Repeat for each sample
    
    #Split dataset in train set (25 sample) and 1 test set
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    
    #Train the model
    rf.fit(x_train_iwt, y_train_iwt)
    
    #Get prediction for untrained data 
    prediction_iwt = rf.predict(x_test_iwt)
      
    #Add specific prediction of all predictions
    rf_pred.append(prediction_iwt)
    
#Get rf accuracy and kappa for iwt
print(accuracy_score(rf_pred, y_iwt))
print(cohen_kappa_score(rf_pred, y_iwt))

#SVC for AB Study
from sklearn import svm
svc = svm.SVC(class_weight = 'balanced')

svc.fit(x_train_ab, y_train_ab)
predictions_ab = svc.predict(x_test_ab)

print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))

#SVC for IWT study
svc_pred = []

for train_index, test_index in loo.split(x_iwt): 
    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    
    svc.fit(x_train_iwt, y_train_iwt)
    
    prediction_iwt = svc.predict(x_test_iwt)
      
    svc_pred.append(prediction_iwt)
    
print(accuracy_score(svc_pred, y_iwt))
print(cohen_kappa_score(svc_pred, y_iwt))

#SGD for AB Study
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(class_weight = 'balanced', loss = "log", 
                    penalty="elasticnet", random_state =  48151)

sgd.fit(x_train_ab, y_train_ab)
predictions_ab = sgd.predict(x_test_ab)

print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))

#SGD for IWT study
sgd_pred = []

for train_index, test_index in loo.split(x_iwt): 
    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    
    sgd.fit(x_train_iwt, y_train_iwt)
    
    prediction_iwt = sgd.predict(x_test_iwt)
      
    sgd_pred.append(prediction_iwt)
    
print(accuracy_score(sgd_pred, y_iwt))
print(cohen_kappa_score(sgd_pred, y_iwt))

#KNN for AB Study
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(x_train_ab, y_train_ab)
predictions_ab = knn.predict(x_test_ab)

print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))

#KNN for IWT study
knn_pred = []

for train_index, test_index in loo.split(x_iwt): 
    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    
    knn.fit(x_train_iwt, y_train_iwt)
    
    prediction_iwt = knn.predict(x_test_iwt)
      
    knn_pred.append(prediction_iwt)
    
print(accuracy_score(knn_pred, y_iwt))
print(cohen_kappa_score(knn_pred, y_iwt))

###Hyperparameter tuning with IWT data

#Example with knn
import joblib

#Define function
def knn_train(x_train, y_train, x_valid, y_valid):
    k_values = np.arange(1, 11, 1)
    best_acc = 0
    for k in k_values:
        knn = KNeighborsClassifier(k)
        knn.fit(x_train, y_train)
        prediction = knn.predict(x_valid)
        current_acc = accuracy_score(prediction, y_valid)
        if current_acc > best_acc:
            best_acc = current_acc
            filename = 'best_knn.sav'
            joblib.dump(knn, filename)
    best_knn = joblib.load('best_knn.sav')
    return best_knn

#Test model
best_knn_pred = []
for train_index, test_index in loo.split(x_iwt):    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    x_train_iwt, x_valid_iwt ,y_train_iwt, y_valid_iwt = \
        train_test_split(x_train_iwt,y_train_iwt, test_size = 0.20, 
                         random_state = 48151)
    best_knn = knn_train(x_train_iwt, y_train_iwt, x_valid_iwt, y_valid_iwt)    
    predictions_iwt = best_knn.predict(x_test_iwt)
    best_knn_pred.append(predictions_iwt)

print(accuracy_score(best_knn_pred, y_iwt))
print(cohen_kappa_score(best_knn_pred, y_iwt))

###Hyperparameter tuning for rf 

def rf_train(x_train, y_train, x_valid, y_valid):  
    nb_trees = np.arange(10, 200, 10)
    best_acc = 0
    for n in nb_trees:
        rf = RandomForestClassifier(n, class_weight = 'balanced',
                                    random_state = 48151)
        rf.fit(x_train, y_train)
        prediction = rf.predict(x_valid)
        current_acc = accuracy_score(prediction, y_valid)
        if current_acc > best_acc:
            best_acc = current_acc
            filename = 'best_rf.sav'
            joblib.dump(rf, filename)
    best_rf = joblib.load('best_rf.sav')
    return best_rf


best_rf_pred = []
for train_index, test_index in loo.split(x_iwt):    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    x_train_iwt, x_valid_iwt ,y_train_iwt, y_valid_iwt = \
        train_test_split(x_train_iwt,y_train_iwt, test_size = 0.20, 
                         random_state = 48151)
    best_rf = rf_train(x_train_iwt, y_train_iwt, x_valid_iwt, y_valid_iwt)    
    predictions_iwt = best_rf.predict(x_test_iwt)
    best_rf_pred.append(predictions_iwt)

print(accuracy_score(best_rf_pred, y_iwt))
print(cohen_kappa_score(best_rf_pred, y_iwt))

###Hyperparameter tuning for svc

def svc_train(x_train, y_train, x_valid, y_valid):
    gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    C = [1,10,100]
    best_acc = 0
    for c in C:
        for n in gamma:
            svc = svm.SVC(kernel="rbf", class_weight = 'balanced', 
                          gamma = n, C = c)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_valid)
            current_acc = accuracy_score(prediction, y_valid)
            if current_acc > best_acc:
                best_acc = current_acc
                filename = 'best_svc.sav'
                joblib.dump(svc, filename)
    best_svc = joblib.load('best_svc.sav')
    return best_svc

best_svc_pred = []
for train_index, test_index in loo.split(x_iwt):    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    x_train_iwt, x_valid_iwt ,y_train_iwt, y_valid_iwt = \
        train_test_split(x_train_iwt,y_train_iwt, test_size = 0.20, 
                         random_state = 48151)
    best_svc = svc_train(x_train_iwt, y_train_iwt, x_valid_iwt, y_valid_iwt)    
    predictions_iwt = best_svc.predict(x_test_iwt)
    best_svc_pred.append(predictions_iwt)

print(accuracy_score(best_svc_pred, y_iwt))
print(cohen_kappa_score(best_svc_pred, y_iwt))

#Hyperparameter tuning for sgd

def sgd_train(x_train, y_train, x_valid, y_valid): 
    lr = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    iterations = np.arange(1,1000,100)
    best_acc = 0
    for i in iterations:
        for n in lr:
            sgd = SGDClassifier(loss = "log", penalty = "elasticnet", alpha = n, 
                                max_iter = i, class_weight = 'balanced',
                                random_state = 48151)
            sgd.fit(x_train, y_train)
            prediction = sgd.predict(x_valid)
            current_acc = accuracy_score(prediction, y_valid)
            if current_acc > best_acc:
                best_acc = current_acc
                filename = 'best_sgd.sav'
                joblib.dump(sgd, filename)
    best_sgd = joblib.load('best_sgd.sav')
    return best_sgd


best_sgd_pred = []
for train_index, test_index in loo.split(x_iwt):    
    x_train_iwt, y_train_iwt, x_test_iwt, y_test_iwt = x_iwt[train_index, :],\
        y_iwt[train_index], x_iwt[test_index, :], y_iwt[test_index]
    x_train_iwt, x_valid_iwt ,y_train_iwt, y_valid_iwt = \
        train_test_split(x_train_iwt,y_train_iwt, test_size = 0.20, 
                         random_state = 48151)
    best_sgd = sgd_train(x_train_iwt, y_train_iwt, x_valid_iwt, y_valid_iwt)    
    predictions_iwt = best_sgd.predict(x_test_iwt)
    best_sgd_pred.append(predictions_iwt)

print(accuracy_score(best_sgd_pred, y_iwt))
print(cohen_kappa_score(best_sgd_pred, y_iwt))

#Artificial neural network with AB graph data

#Implement ann_ab
import tensorflow as tf
tf.random.set_seed(48151)

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

#Create artificial neural network
ann_ab = Sequential()

#Add first hidden layer
ann_ab.add(Dense(6, input_shape=(8, ), activation = "relu"))

#Output layer
ann_ab.add(Dense(1, activation='sigmoid'))  

#Hyperparameters
ann_ab.compile(loss='binary_crossentropy')

#Train model
ann_ab.fit(x_train_ab, y_train_ab, epochs = 20)

#Get predictions on test set
predictions_ab = np.round(ann_ab.predict(x_test_ab))

#Return accuracy and kappa
print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))

#Artificial neural network with VS 

x_train_vs, x_test_vs, y_train_vs, y_test_vs =\
    train_test_split(x_vs, y_vs, test_size = 0.20, random_state = 48151)

ann_vs = Sequential()
ann_vs.add(Flatten(input_shape=(10, 26)))
ann_vs.add(Dense(128, activation='relu'))
ann_vs.add(Dense(1, activation='sigmoid'))  
ann_vs.compile(loss='binary_crossentropy')

ann_vs.fit(x_train_vs, y_train_vs, epochs = 20)

predictions_vs = np.round(ann_vs.predict(x_test_vs))

print(accuracy_score(predictions_vs, y_test_vs))
print(cohen_kappa_score(predictions_vs, y_test_vs))

##Hyperparameter tuning epochs for AB graph data

#Split validation set
x_train_ab, x_valid_ab, y_train_ab, y_valid_ab =\
    train_test_split(x_train_ab, y_train_ab, test_size = 0.20, \
                     random_state = 48151)

#Create callback
checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath= './best_ann.hdf5', monitor='val_loss', mode ='min',
        save_best_only=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                      mode='min', patience = 20)

#Train data
ann_ab.fit(x_train_ab, y_train_ab, epochs = 500,
        validation_data = (x_valid_ab, y_valid_ab), 
        callbacks=[checkpointer, es])    

#Get prediction of best model
best_ann = tf.keras.models.load_model('./best_ann.hdf5', compile=True)
predictions_ab = np.round(best_ann.predict(x_test_ab))

print(accuracy_score(predictions_ab, y_test_ab))
print(cohen_kappa_score(predictions_ab, y_test_ab))
