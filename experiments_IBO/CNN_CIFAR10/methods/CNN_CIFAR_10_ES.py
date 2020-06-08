from __future__ import print_function
import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pickle
import time
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

import logging
logging.basicConfig(level=logging.INFO)


from robo.fmin import entropy_search
from robo.fmin import bayesian_optimization

from importance_sampling.training import ImportanceTraining

import os
   
def objective_function(param):
    start_time = time.time()
    nb_epochs = 50

    
    # transform the input to the real range of the hyper-parameters, to be used for model training
    batch_size = int(param[0])
    learning_rate = param[1]
    learning_rate_decay = param[2]
    l2_regular = param[3]
    conv_filters =int( param[4])
    dense_units= int(param[5])

    print("[parameters: batch_size: {0}/lr: {1}/lr_decay: {2}/l2: {3}/conv_filters: {4}/dense_unit: {5}]".format(\
        batch_size, learning_rate, learning_rate_decay, l2_regular, conv_filters, dense_units))

    num_conv_layers = 3
    dropout_rate = 0.0
    kernel_size = 5
    pool_size = 3
    
    
    # build the CNN model using Keras
    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                     input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    if num_conv_layers >= 3:
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=learning_rate_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,batch_size=batch_size,epochs=nb_epochs,validation_data=(x_test, y_test), verbose=1)
    #ImportanceTraining(model,presample=6).fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs,validation_data=(x_test, y_test))
    loss, score = model.evaluate(x_test, y_test, verbose=1)
    print('Loss:',loss)
    print('Val error:', 1.0 - score)
    c = time.time() - start_time
    print('Loss:',loss)
    print('Test error:', 1.0 - score)
    
    global obj_track
    obj_track.append( 1- score)
    global loss_track
    loss_track.append(loss)
    print("Obj Track: ",obj_track)
    print("Loss Track: ",loss_track)
    global iter_num
    iter_num = iter_num+1
    print("Iter num:",iter_num)
    global best_obj
    best_obj = min(best_obj, 1- score)
    global best_loss
    best_loss = min(best_loss, loss)
    print("Best Error: ",best_obj)
    print("Best Loss:",best_loss)
    print("#######################")
    end_time = time.time()- start_time
    print("Time to run this hp:",end_time)
    print("#######################")
    return 1.0 - score
##################################################################################
# Main Script
##################################################################################
#########################
# setup ES function        
#########################
home_dir = '/home'
######################
# load data          
######################

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#################################################
# setting hp bounds        
#################################################
lower = np.array([32,1e-7,1e-7,1e-7,128,256])
upper = np.array([512,0.1,1e-3,1e-3,256,512])

#################################################
# Book keeping setup  
#################################################
exp_name = 'CNN_CIFAR10'
method ='ES'
n_runs = 1
best_obj=1000
best_loss=1000
obj_track=[]
loss_track=[]
iter_num =0
results_over_runs = dict()
#################################################
# Initial data setup
#################################################
version =5
n_init_num=5
budget_iter=140
x_init_dir=home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/initial_data/'
x_init_name = 'x_init_{}_v{}.pkl'.format(exp_name,version)

#################################################
# IBO main function      
#################################################
for it in range(n_runs):
    results_over_runs[it] = entropy_search(objective_function, lower=lower, upper=upper,init_data=x_init_dir+x_init_name,num_iterations=budget_iter)
#################################################
# Saving the results
#################################################
output_main_dir =home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/output_main/'
pickle.dump( results_over_runs, open(output_main_dir+"results_{}_{}_init{}_budget{}_v{}.pkl".format(exp_name,method,n_init_num,budget_iter,version), "wb" ))

