#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:25:04 2023

@author: tavo
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import preprocessing as pr
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Activation, Dense, Layer, BatchNormalization

###############################################################################
# Loading packages 
###############################################################################

seed=762

np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

batchSize = 512
epochs = 5
nsplits = 10

###############################################################################
# Loading packages 
###############################################################################

def TrainModelKfold(makemodel,data,featlabels,targetlabels,lr,n_splits=2,loss='mse',AE=True,scaled=True):
    
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=1098)
    
    dataindex = data.index
    modelcontainer = []
    scalers = []

    for i, (train_index, test_index) in enumerate(kf.split(dataindex)):
    
        loopTrainIndex = dataindex[train_index]
        loopTestIndex = dataindex[test_index]
        
        xtrain = data[featlabels].loc[loopTrainIndex].values
        xtest = data[featlabels].loc[loopTestIndex].values
        
        if scaled:
            scaler = pr.MinMaxScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
            scalers.append(scaler)
        
        if AE:
            ytrain = xtrain
            ytest = xtest            
        else:    
            ytrain = data[targetlabels].loc[loopTrainIndex].values
            ytest = data[targetlabels].loc[loopTestIndex].values
        
        print('Fold' + str(i))
        
        model = makemodel()        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                      loss=loss)
        model.fit(xtrain, ytrain,
                  validation_data=(xtest, ytest), 
                  batch_size=batchSize, 
                  epochs=epochs)
        
        modelcontainer.append(model)
        
    return scalers,modelcontainer
    
    
###############################################################################
# Loading packages 
###############################################################################

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*(0.0001)*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        self.add_metric(klbatch,name='kl_loss',aggregation='mean')
        
        return inputs

class Sampling(Layer):
    '''
    Custom sampling layer
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}
    
    @tf.autograph.experimental.do_not_convert   
    def call(self,inputs,**kwargs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon

class DynamicsLayer(Layer):
    '''
    Custom Dynamics loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.is_placeholder=True
        super(DynamicsLayer,self).__init__(*args,**kwargs)
    
    def build(self,input_shape,*args,**kwargs):
        
        w_init = tf.random_normal_initializer()
        b_init = tf.random_normal_initializer()
        
        self.w = tf.Variable(initial_value=w_init(shape=(2, 3),
                             dtype='float32'),trainable=True)
        
        self.b = tf.Variable(initial_value=b_init(shape=(2,),
                             dtype='float32'),trainable=True)
        

    @tf.autograph.experimental.do_not_convert 
    def call(self,inputs,*args,**kwargs):
        
        xcoords = inputs[:,0]
        ycoords = inputs[:,1]
        dxdt = xcoords[1:]-xcoords[:-1]
        dydt = ycoords[1:]-ycoords[:-1]
        ddt = tf.stack([dxdt,dydt])
        
        vector = tf.stack([inputs[:,0],
                           inputs[:,1],
                           tf.multiply(inputs[:,0],inputs[:,1])
                           ])
        
        product = tf.matmul(self.w,vector)
        shape = tf.shape(product)
        
        constants = tf.reshape(tf.repeat(self.b, shape[1]-1,axis=-1),[2,shape[1]-1])
        product = product[:,0:-1] + constants
  
        dyloss = (0.0001)*tf.math.squared_difference(ddt,product)
        dyloss = K.mean(dyloss) #+ (10**-6)*tf.norm(self.w,ord=1)
        self.add_loss(dyloss,inputs=inputs)
        self.add_metric(dyloss,name='dynamics_loss',aggregation='mean')
        
        return inputs
    
###############################################################################
# Loading packages 
###############################################################################

def EncoderModel():
    
    InputFunction = Input(shape=(340,))
    
    X = Dense(2,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    Mu = Dense(2)(X)
    LogSigma = Dense(2)(X)
    Mu,LogSigma = KLDivergenceLayer()([Mu,LogSigma])
    Samp = Sampling()([Mu,LogSigma])
    Output = DynamicsLayer()(Samp)
    encoder = Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction, encoder

def DecoderModel():
    
    InputFunction = Input(shape=(2,))
    
    X = Dense(2,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(340,use_bias=False)(X)
    X = BatchNormalization()(X)
    Output = Activation('sigmoid')(X)
    
    decoder = Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction, decoder

def VAEModel():
    
    InputEncoder,Encoder = EncoderModel()
    InputDecoder,Decoder = DecoderModel()
    Output = Decoder(Encoder(InputEncoder))
    VAE = Model(inputs=InputEncoder,outputs=Output)
    
    return VAE

###############################################################################
# Loading packages 
###############################################################################

data0 = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv')
data1 = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')

data0 = data0.set_index('id')
data1 = data1.set_index('id')

finalindex = data0.index.intersection(data1.index)

data0 = data0.loc[finalindex]
data1 = data1.loc[finalindex]

###############################################################################
# Loading packages 
###############################################################################

inxTrain, inxTest,_,_ = train_test_split(data0.index, data0.index,
                                         test_size=0.1, random_state=42)

stepsperepoch = (inxTrain.shape[0] - (inxTrain.shape[0]//nsplits))//batchSize

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(0.01,
                                                                  stepsperepoch//2,
                                                                  t_mul=1.5,
                                                                  m_mul=0.45,
                                                                  alpha=0.01)


steps = epochs*((inxTrain.shape[0] - (inxTrain.shape[0]//5))//batchSize)

rates = [lr_decayed_fn.__call__(val).numpy() for val in range(steps)]

plt.plot(rates)

scalersContainer,vaeContainer = TrainModelKfold(VAEModel,
                                                data0.loc[inxTrain],
                                                data0.columns,
                                                data0.columns,
                                                lr_decayed_fn,
                                                n_splits=nsplits)

###############################################################################
# Loading packages 
###############################################################################

fig,axs = plt.subplots(nsplits,1,figsize=(10,20))
k = 0
perfs = []
minidata = data0.loc[inxTest].values
for sc,mod in zip(scalersContainer,vaeContainer):
    loopdata = sc.transform(minidata)
    contdata = mod.layers[1].predict(loopdata)
    perfs.append(mod.evaluate(loopdata))
    axs[k].scatter(contdata[:,0],contdata[:,1],alpha=0.025)
    k = k+1
    

###############################################################################
# Loading packages 
###############################################################################
'''
selected = -1

selectedScaler = scalersContainer[selected]
selectedModel = vaeContainer[selected]

selectedModel.layers[2].save('/media/tavo/storage/model/trained/decoder.h5')

###############################################################################
# Loading packages 
###############################################################################

feats = selectedModel.layers[1].predict(selectedScaler.transform(data0.values))

datafeats = pd.DataFrame()

datafeats['id'] = data0.index
datafeats['feat0'] = feats[:,0]
datafeats['feat1'] = feats[:,1]

datafeats.to_csv('/media/tavo/storage/model/trained/encoded.csv',index=False)
'''