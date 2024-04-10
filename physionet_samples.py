#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from utils import *


# In[2]:


#Get all file names
path = '../your_path/set-a'
csv_files = glob.glob(path + "/*.txt")
outcomes=pd.read_csv(path +'/Outcomes.csv')


# In[3]:


print(outcomes.head(3))


# In[4]:


#Load in csv format and concat all files
data = (pd.read_csv(file) for file in csv_files)
data   = pd.concat(data, ignore_index=True)
print(data.head(2))


# In[5]:


# data.Parameter.unique()


# In[6]:


univariate_length=data.groupby(['Parameter','RecordID'])['Value'].count().values
univariate_length=int(np.max(univariate_length))
print(f'# of observations per univariate time series {univariate_length}')


# In[7]:


no_missing_values=len(data.loc[data.Value==-1,:])
print(f'# of missing values {no_missing_values}')


# In[8]:


#Create mask that determines whether a value is missing.
data['Mask']=data.Value.apply(lambda x: 1 if x>=0 else 0)


# In[9]:


#Construction of samples
VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES,OUTCOMES_2,PATIENT_ID=samplesConstructorPhysionet(data,outcomes,univariate_length)


# In[24]:


#Save samples
np.save(path+'/VALUES',VALUES)
np.save(path+'/TIMESTAMPS',TIMESTAMPS)
np.save(path+'/MASKS',MASKS)
np.save(path+'/PADDINGS',PADDINGS)
np.save(path+'/OUTCOMES',OUTCOMES)
np.save(path+'/OUTCOMES_2',OUTCOMES_2)
np.save(path+'/PATIENT_ID',PATIENT_ID)

