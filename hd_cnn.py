# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:37:12 2021

@author: tboonesifuentes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Activation, Lambda, Conv2D, MaxPool2D, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG19
import metrics as metrics
import numpy as np
import models as models
import pandas as pd
import seaborn as sns
from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from data import Birds
from sklearn.metrics import accuracy_score
from numpy import save,load


#Load dataset

dataset=Birds()
#u = dataset.draw_taxonomy()
#u.view()

#Set variables

epochs=1
batch=100


num_classes = [dataset.num_classes_l0, dataset.num_classes_l1, dataset.num_classes_l2]

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
performance_callback = models.performance_callback(dataset.X_val,dataset.y_val,dataset.taxonomy)


def HD_CNN(coarse_lv,fine_lv,epochs_c,epochs_f):
    
    #coarse_lv: (int) number of coarse classes. Should be less than fine_lv
    #fine_lv: (int) number of fine classes. Should be more than coarse_lv
    # epochs_c: number of epochs for coarse model
    # epochs_f: number of epochs for fine models

    coarse_categories = num_classes[coarse_lv]
    fine_categories = num_classes[fine_lv]
    
    y_c_test=dataset.y_test[coarse_lv]
    y_f_test=dataset.y_test[fine_lv]
    
    fine2coarse = np.zeros((fine_categories,coarse_categories))
    for i in range(coarse_categories):
        index = np.where(y_c_test == i)[0]
        fine_cat = np.unique([y_f_test[j] for j in index])
        for j in fine_cat:
            fine2coarse[j,i] = 1
    
    #get coarse classifier
    coarse_model=models.get_hdcnn_coarse_classifier(num_classes, dataset.image_size,level=coarse_lv)
    
    coarse_model.summary()
    
    #Training coarse model
    coarse_model.fit(dataset.X_train, 
                     dataset.y_train[coarse_lv],
                     validation_data = (dataset.X_val, dataset.y_val[coarse_lv]),
                     batch_size=batch, 
                     epochs=epochs_c,
                     callbacks=[early_stopping_callback])
    
   
    
    for i in range(coarse_categories):
        
        # Get all training data for the coarse category
        
        y_train_hot=metrics.one_hot(dataset.y_train[fine_lv])

        #get train labels that are inside each coarse classifier
        ix = np.where([(y_train_hot[:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
        x_tix = dataset.X_train[ix]
        y_train = dataset.y_train[fine_lv]
        y_tix = y_train[ix]
        
        #Definea model for each fine classifier
        fine_model = models.get_hdcnn_fine_classifiers(num_classes, dataset.image_size,level=fine_lv)
       
    
        fine_model.fit(x_tix, y_tix, 
                       batch_size=batch, 
                       epochs=epochs_f,
                       callbacks=[early_stopping_callback]
                       )
        
        fine_model.save_weights('fine_models/fine_model'+str(i))
        
   
#Evaluation mode
        
    y_test_hot=metrics.one_hot(dataset.y_test[fine_lv])
        
    yh = np.zeros(np.shape(y_test_hot))
    
    #evaluate coarse model
    yh_c = coarse_model.predict(dataset.X_test)
    
    for i in range(coarse_categories):
        
        #Coarse predictions
        
        print("Evaluating Fine Classifier: ", str(i))
        
        fine_model.load_weights('fine_models/fine_model'+str(i))
        
        #evaluate fine models
        fine_predictions=fine_model.predict(dataset.X_test)
        
        #calculate final predictions
        yh += np.multiply(yh_c[:,i].reshape((len(dataset.y_test[fine_lv])),1),fine_predictions)
    

    yh_index=np.argmax(yh,axis=1)
    yh_c_index=np.argmax(yh_c,axis=1)
    #acc= accuracy_score(dataset.y_test[fine_lv],yh_index)
    
    return yh_c_index,yh_index


yhc1,yh1 = HD_CNN(0,1,2,2)

#filename='c_pred2.npy'
#save(filename, c_pred2)
#filename='f_pred2.npy'
#save(filename, f_pred2)
#filename='acc2.npy'
#save(filename, acc2)

#c_pred1,f_pred1,acc1 = HD_CNN(0,1,10,10)

#c_pred2,f_pred2,acc2 = HD_CNN(1,2,3,3)

#c_pred3,f_pred3,acc3 = HD_CNN(2,3,3,3)

#c_pred4,f_pred4,acc4 = HD_CNN(3,4,3,3)

#for j in range(len(num_classes)-1):
#    
#    c_pred,f_pred,acc = HD_CNN(j,(j+1),3,3)
#    
#    filename='c_pred'+'_'+str(j+1)+'.npy'
#    save(filename, c_pred)
#    filename='f_pred'+'_'+str(j+1)+'.npy'
#    save(filename, f_pred)
#    filename='acc'+'_'+str(j+1)+'.npy'
#    save(filename, acc)
#    del c_pred,f_pred,acc
#    gc.collect()





