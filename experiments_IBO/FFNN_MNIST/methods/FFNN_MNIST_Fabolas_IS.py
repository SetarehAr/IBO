import os
import sys
import time
import numpy as np
import pickle 

import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path

from robo.fmin import fabolas
from keras.datasets import mnist
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam,RMSprop

from importance_sampling.training import ImportanceTraining


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def build_model(x):
    
    first_input=784
    last_output=10
        
    l1_drop = float(x[0]) #==> prob of dropout
    l1_out = int(np.exp(float(x[1])))  #==> number of nodes in hidden layers
    batchsize = int(np.exp(float(x[2]))) #==> batch size
    n_hidden_layers = int(x[3]) #==> number of hidden layers
    learningrate = np.exp(float(x[4]))  #==> learning rate
    decayrate =  float(np.exp(float(x[5])))  #==> decay rate

    model = Sequential()
    model.add(Dense(l1_out, input_shape=(first_input,)))
    model.add(Activation('relu'))
    model.add(Dropout(l1_drop))
    
    for i in range(n_hidden_layers - 1):
        model.add(Dense(l1_out))
        model.add(Activation( 'relu'))
        model.add(Dropout(l1_drop))
    
    opt = RMSprop(lr=learningrate, decay=decayrate)
    


    model.add(Dense(last_output))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model


def evaluate_error_model(X_tr_loc, Y_tr_loc, X_val, Y_val, params,pre_sample):
    start_time= time.time()
    batchsize = int(np.exp(float(params[2]))) #==> batch size

    model = build_model(params)
    
    nb_epochs = 2
    ImportanceTraining(model, presample=pre_sample).fit(X_tr_loc, Y_tr_loc,batch_size=batchsize,epochs=nb_epochs,verbose=1, validation_data=(X_val, Y_val))

    loss, score = model.evaluate(X_val, Y_val, batch_size=batchsize, verbose=1)
    print('Val error:', 1.0 - score)
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


def objective_function(params, s):
    s = int(s) # s is not transformed in range [0 1] at this step
    start_time = time.time()

    # Shuffle the data and split up the request subset of the training data
    s_max = Y_train.shape[0]
    shuffle = np.random.permutation(np.arange(s_max))
    train_subset = X_train[shuffle[:s]]
    train_targets_subset = Y_train[shuffle[:s]] 

    B_min =2
    B_max=6
    global initial
    if initial < B_max:
        B= B_max
    else:
        B= np.random.uniform(low=B_min,high=B_max)
        if B*params[2]>s:
            print("The big batch size for importance sampling is larger than the training data size.")
            B= B_min
    initial=initial+1

    B=6
    y=evaluate_error_model(train_subset,train_targets_subset,X_test,Y_test,params,float(B))
    c = time.time() - start_time

    return y, c

##################################################################################
# Main Script
##################################################################################
#########################
# setup IBO function        
#########################
home_dir = '/home/'
######################
# load data          
######################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)  

#################################################
# setting max and min number of training samples        
#################################################
s_min = 256 # maximum batch size for MNIST
s_max = 60000 # entire training data
#################################################
# setting hp bounds        
#################################################
lower = np.array([0,np.log(16),np.log(8),1,np.log(1e-7),1e-7])
upper = np.array([0.5,np.log(256),np.log(256),5,np.log(0.1),1e-3])
#################################################
# Book keeping setup  
#################################################
exp_name = 'FFNN_MNIST'
method= 'Fabolas_IS'
n_runs = 1
best_obj=1000
best_loss=1000
obj_track=[]
loss_track=[]
iter_num =0
initial=0
results_over_runs = dict()
#################################################
# Initial data setup
#################################################
version =1
n_init_num=5
budget_iter=7
x_init_dir=home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/initial_data/'
x_init_name = 'x_init_{}_v{}.pkl'.format(exp_name,version)

#################################################
# Fabolas main function  (with IS-based objective function)    
#################################################
for it in range(n_runs):
    results_over_runs[it] = fabolas(objective_function, lower=lower, upper=upper,s_min=s_min, s_max=s_max,init_data=x_init_dir+x_init_name,subsets=[1],num_iterations=budget_iter)

output_main_dir =home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/output_main/'
pickle.dump( results_over_runs, open(output_main_dir+"results_{}_{}_init{}_budget{}_v{}.pkl".format(exp_name,method,n_init_num,budget_iter,version), "wb" ))
print(results_over_runs[0])

