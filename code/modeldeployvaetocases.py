#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:45:26 2024

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

datac = pd.read_csv('/media/tavo/storage/01localApp/model/archive/selecteddata.csv')

datac['date'] = pd.to_datetime(datac['date'])
datac['year'] = datac['date'].dt.year.astype(np.float32)

datac['week'] = datac['date'].dt.isocalendar().week.astype(np.float32)
datac['dayofweek'] = datac['date'].dt.dayofweek.astype(np.float32)
datac['month'] = datac['date'].dt.month.astype(np.float32)

datac['semester'] = np.where(datac['date'].dt.quarter.gt(2),2,1)

maxs = datac.groupby(['qry','year','semester'])['cases'].max()

tups= [tuple([val,sal,xal]) for val,sal,xal in zip(datac['qry'],datac['year'],datac['semester'])]

datac['maxs'] = maxs.loc[tups].values
datac['normcases'] = datac['cases']/datac['maxs']

datac = datac[(datac['maxs']>150) & (datac['maxs']<25000)]

windowsize = 28
perlocationts = datac.groupby(['dayofyear','qry'])['normcases'].mean().rolling(windowsize,min_periods=1).mean().unstack()
perlocationts = (perlocationts - perlocationts.min())/(perlocationts.max() - perlocationts.min())
perlocationts = perlocationts.stack()

tupsl = [tuple([val,sal]) for val,sal in zip(datac['dayofyear'],datac['qry'])]
datac['rollingnormcases'] =  perlocationts.loc[tupsl].values

feats = ['dayofyear','week','month','lat','long','spots','lengthofday',
         'CloudFrc_A','TotO3_A']

yearlycases = datac.groupby(feats)['rollingnormcases'].mean()
yearlycases = yearlycases.reset_index()


del datac

###############################################################################
# Loading packages 
###############################################################################

decoderModel = tf.keras.models.load_model('/media/tavo/storage/model/trained/decoder.h5')
toLatentModel = tf.keras.models.load_model('/media/tavo/storage/model/trained/featsToLatent.h5')

###############################################################################
# Loading packages 
###############################################################################

LatentData = toLatentModel.predict(yearlycases[feats].values)
compositionData = decoderModel.predict(LatentData)

compositionData = pd.DataFrame(compositionData)

compositionData['cases'] = yearlycases['rollingnormcases'].values

for k,val in enumerate(feats):
    compositionData[val] = yearlycases[val].values

fcolumns = [k for k in range(340)] + feats

###############################################################################
# Loading packages 
###############################################################################

def ToLearnedModel():
    
    InputFunction = Input(shape=(349,))
    
    X = Dense(256,use_bias=False)(InputFunction)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(128,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(64,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(32,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(16,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(8,use_bias=False)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Dense(1)(X)
    X = BatchNormalization()(X)
    Output = Activation('sigmoid')(X)
    
    outModel = Model(inputs=InputFunction,outputs=Output)

    return outModel

###############################################################################
# Loading packages 
###############################################################################

inxTrain, inxTest,_,_ = train_test_split(compositionData.index, compositionData.index,
                                         test_size=0.1, random_state=42)

stepsperepoch = (inxTrain.shape[0] - (inxTrain.shape[0]//nsplits))//batchSize

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(0.01,
                                                                  stepsperepoch//2,
                                                                  t_mul=1.5,
                                                                  m_mul=0.45,
                                                                  alpha=0.01)

scalersContainer,vaeContainer = TrainModelKfold(ToLearnedModel,
                                                compositionData.loc[inxTrain],
                                                fcolumns,
                                                'cases',
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
    perfs.append(mod.evaluate(x=compositionData[fcolumns].loc[inxTest].values,y=compositionData['cases'].loc[inxTest].values))

selectedModel = vaeContainer[np.argmin(perfs)]

selectedModel.save('/media/tavo/storage/model/trained/compositionToCases.h5')

###############################################################################
# Loading packages 
###############################################################################

preds = selectedModel.predict(compositionData[fcolumns].loc[inxTest].values)

plt.figure()
plt.scatter(compositionData['cases'].loc[inxTest].values.ravel(),preds.ravel(),alpha=0.01)
