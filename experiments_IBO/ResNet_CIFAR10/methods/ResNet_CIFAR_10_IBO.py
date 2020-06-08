
from __future__ import print_function

import time

from keras import backend as K
from keras.callbacks import LearningRateScheduler, Callback
from keras.datasets import cifar10,cifar100
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, \
    GlobalAveragePooling2D, Input, add, Convolution2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
import numpy as np

from importance_sampling.layers.normalization import LayerNormalization
from importance_sampling.datasets import CIFAR10,CIFAR100,ZCAWhitening
from importance_sampling.training import ImportanceTraining
import random
from scipy.io import loadmat

from robo.fmin import IBO
from importance_sampling.training import ImportanceTraining

import pickle 

import os

        
def wide_resnet(L, k, l2_reg,drop_rate=0.0):
    """Implement the WRN-L-k from 'Wide Residual Networks' BMVC 2016"""
    
    def wide_resnet_impl(input_shape, output_size):
        def conv(channels, strides,
                 params=dict(padding="same", use_bias=False,
                             kernel_regularizer=l2(l2_reg))):
            def inner(x):
                x = LayerNormalization()(x)
                x = Activation("relu")(x)
                x = Convolution2D(channels, 3, strides=strides, **params)(x)
                x = Dropout(drop_rate)(x) if drop_rate > 0 else x
                x = LayerNormalization()(x)
                x = Activation("relu")(x)
                x = Convolution2D(channels, 3, **params)(x)
                return x
            return inner

        def resize(x, shape):
            if K.int_shape(x) == shape:
                return x
            channels = shape[3 if K.image_data_format() == "channels_last" else 1]
            strides = K.int_shape(x)[2] // shape[2]
            return Convolution2D(
                channels, 1, padding="same", use_bias=False, strides=strides
            )(x)

        def block(channels, k, n, strides):
            def inner(x):
                for i in range(n):
                    x2 = conv(channels*k, strides if i == 0 else 1)(x)
                    x = add([resize(x, K.int_shape(x2)), x2])
                return x
            return inner

        # According to the paper L = 6*n+4
        n = int((L-4)/6)

        group0 = Convolution2D(16, 3, padding="same", use_bias=False,
                               kernel_regularizer=l2(l2_reg))
        group1 = block(16, k, n, 1)
        group2 = block(32, k, n, 2)
        group3 = block(64, k, n, 2)

        x_in = x = Input(shape=input_shape)
        x = group0(x)
        x = group1(x)
        x = group2(x)
        x = group3(x)

        x = LayerNormalization()(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(output_size, kernel_regularizer=l2(l2_reg))(x)
        y = Activation("softmax")(x)

        model = Model(inputs=x_in, outputs=y)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        return model
    return wide_resnet_impl

class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
 
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        
class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=40):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
 
    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
 
        # return the learning rate
        return float(alpha)

   
def objective_function(params,B):
    nb_epochs=50
    start_time = time.time()
    depth = 28
    width =2
    batch_size = 128
    

    lr_initial_hp = float(np.exp(params[0]))
    decay_rate_factor_hp = float(np.exp(params[1]))
    l2_regular_weight_hp = float(np.exp(params[2]))
    momentum_hp = float(params[3]) 
    
    model = wide_resnet(depth, width,l2_regular_weight_hp )(dset.shape, dset.output_size)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=lr_initial_hp, momentum=momentum_hp),
        metrics=["accuracy"]
    )

    
    # Create the data augmentation generator
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)
    datagen.fit(x_train)
    
    
    schedule = StepDecay(initAlpha=lr_initial_hp, factor=decay_rate_factor_hp, dropEvery=40)
    callbacks = [LearningRateScheduler(schedule)]
    

    ImportanceTraining(model,presample=float(B)).fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),validation_data=(x_test, y_test),epochs=nb_epochs, verbose=1,batch_size=batch_size,callbacks=callbacks,steps_per_epoch=int(np.ceil(float(len(x_train)) / batch_size)))
    
    loss, score = model.evaluate(x_test, y_test, verbose=1)
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
    return 1.0 - score, c
##################################################################################
# Main Script
##################################################################################
#########################
# setup IBO function        
#########################
home_dir = '/home'
######################
# load data          
######################

dset = ZCAWhitening(CIFAR10())
x_train, y_train = dset.train_data[:]
x_test, y_test = dset.test_data[:]

#################################################
# setting max and min number of training samples        
#################################################
B_min = 2 # maximum batch size 
B_max = 6 # entire training data
#################################################
# setting hp bounds        
#################################################
lower = np.array([np.log(10**-6),np.log(10**-4),np.log(10**-6),0.1])
upper = np.array([np.log(1),np.log(1),np.log(1),0.999])

#################################################
# Book keeping setup  
#################################################
exp_name = 'ResNet_CIFAR10'
method ='IBO'
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
version =3
n_init_num=5
budget_iter=50
x_init_dir=home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/initial_data/'
x_init_name = 'x_init_{}_v{}.pkl'.format(exp_name,version)

#################################################
# IBO main function      
#################################################
for it in range(n_runs):
    results_over_runs[it] = IBO(objective_function, lower=lower, upper=upper,s_min=B_min, s_max=B_max,init_data=x_init_dir+x_init_name,subsets=[1],num_iterations=budget_iter)
#################################################
# Saving the results
#################################################
output_main_dir =home_dir+'/IBO_master/experiments_IBO/'+exp_name+'/output_main/'
pickle.dump( results_over_runs, open(output_main_dir+"results_{}_{}_init{}_budget{}_v{}.pkl".format(exp_name,method,n_init_num,budget_iter,version), "wb" ))

print(results_over_runs[0])
