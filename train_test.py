#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import sklearn
from sklearn.metrics import average_precision_score,roc_curve,auc,f1_score,confusion_matrix
from sklearn.model_selection import KFold,StratifiedKFold
from collections import Counter
from utils import *
import time as tm
from models import *


# In[ ]:


#Load samples
data_set='mimic3'
path=f'data/{data_set}'

if data_set=='physionet':
    Values=np.transpose(np.load(path+'/Values.npy'),[0,2,1])
    TimeStamps=np.transpose(np.load(path+'/Timestamps.npy'),[0,2,1])
    Masks=np.transpose(np.load(path+'/Masks.npy'),[0,2,1])
    PADDINGS=np.transpose(np.load(path+'/Padds.npy'),[0,2,1])
    Target=np.load(path+'/Targets.npy')
else:
    Values=np.load(path+'/Values.npy')
    TimeStamps=np.load(path+'/Timestamps.npy')
    Masks=np.load(path+'/Masks.npy')
    PADDINGS=np.load(path+'/Padds.npy')
    Target=np.load(path+'/Targets.npy')


# In[ ]:


#Compute the empirical mean
VALUES_tempo=np.nan_to_num(Values,nan=0)
means=np.sum(np.sum(VALUES_tempo,1),0)/np.sum(np.sum(Masks,axis=1),0)

if data_set=='mimic3':
    #Replace the average with the mode, because the feature at index 9 is categorical.
    means[9]=1

#Expand means size for imputation
means=means.reshape(1,1,Values.shape[2])
means=np.tile(means,(Values.shape[0],Values.shape[1],1))

#Replace by the correspnding mean NAN/Negative value in VALUES matrix if any.
Values=np.where(np.isnan(Values),means,Values)
Values=np.where(Values<0,means,Values)


# In[ ]:


#Creation of the masl use for loss reconstruction
#We sue 10 percent of observed values
percentage=10
MASKS_IMP=maskImputation(percentage,Masks)
#Distribution of observed values and missing ones
print(Counter(MASKS_IMP.reshape(-1)))
print(MASKS_IMP.shape)


# In[ ]:


tf.random.set_seed(14)
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=14)
bc=tf.keras.losses.BinaryCrossentropy(from_logits=False)
start= tm.perf_counter()
aucs,aucprs=[],[]

observation_period=48
epoch=65
batch_size=1000
time_ref_parameter=1

for train, test in kfold.split(Values,Target):
    model=ALNN_VARIANT(max_time=observation_period,time_interval=time_ref_parameter,
                   gru_unit=168,gru_dropout=0.0,pseudo_latent_dropout=0.0,type_distance="abs",
                   alnn_dropout=0.03,rate_dropout_wcl=0.0,score_loss=True)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])

    TRAINING_SET=[Values[train],TimeStamps[train],Masks[train],MASKS_IMP[train],PADDINGS[train]]
    model.fit(TRAINING_SET,Target[train],verbose=1,batch_size=batch_size,epochs=epoch)


    TESTING_SETS=[Values[test],TimeStamps[test],Masks[test],MASKS_IMP[test],PADDINGS[test]]
    loss_test, accuracy_test = model.evaluate(TESTING_SETS,Target[test],verbose=1,batch_size=100)

    y_probas = model.predict(TESTING_SETS).ravel()
    fpr,tpr,thresholds=roc_curve(Target[test],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC= ',auc(fpr,tpr))

    auprc_ = average_precision_score(Target[test], y_probas)
    aucprs.append(auprc_)
    print('AUPRC= ', auprc_)


finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print(f'Scores with the {prior_hours}-prior hor data:')
print(f'AUC: mean={np.round(np.mean(np.array(aucs)),3)},std={np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean={np.round(np.mean(np.array(aucprs)),3)},std={np.round(np.std(np.array(aucprs)),3)}')
print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




