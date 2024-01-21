#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:05:21 2024

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
epochs = 50
nsplits = 7

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

data0 = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
data1 = pd.read_csv('/media/tavo/storage/model/trained/encoded.csv')

data0 = data0.set_index('id')
data1 = data1.set_index('id')

finalindex = data0.index.intersection(data1.index)

data0 = data0.loc[finalindex]
data1 = data1.loc[finalindex]

#data1 = (data1 - data1.min())/(data1.max() - data1.min())

data = pd.concat([data0,data1],axis=1)
data['date'] = pd.to_datetime(data['date'])
data['week'] = data['date'].dt.isocalendar().week.astype(np.float32)
data['dayofweek'] = data['date'].dt.dayofweek.astype(np.float32)
data['month'] = data['date'].dt.month.astype(np.float32)

feats = ['dayofyear','week','month','lat','lon','sunspots','daylength',
         'CloudFrc_A','TotO3_A']

targets = ['feat0','feat1']

###############################################################################
# Loading packages 
###############################################################################

def ToLearnedModel():
    
    InputFunction = Input(shape=(len(feats),))
    
    X = Dense(256,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(128,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(32,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(4,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    Output = Dense(2)(X)
    
    outModel = Model(inputs=InputFunction,outputs=Output)

    return outModel

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

scalersContainer,vaeContainer = TrainModelKfold(ToLearnedModel,
                                                data.loc[inxTrain],
                                                feats,
                                                targets,
                                                lr_decayed_fn,
                                                n_splits=nsplits,
                                                loss='mse',
                                                AE=False,
                                                scaled=False)

###############################################################################
# Loading packages 
###############################################################################

perfs = []
for mod in vaeContainer:
    perfs.append(mod.evaluate(x=data[feats].loc[inxTest].values,y=data[targets].loc[inxTest].values))

selectedModel = vaeContainer[np.argmin(perfs)]

selectedModel.save('/media/tavo/storage/model/trained/featsToLatent.h5')

###############################################################################
# Loading packages 
###############################################################################

preds = selectedModel.predict(data[feats].loc[inxTest].values)

plt.figure()
plt.scatter(data[targets].loc[inxTest].values.ravel(),preds.ravel(),alpha=0.01)
