import numpy as np
from collections import Counter
import copy
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm


def samplesConstructorPhysionet(df,outcomes,univariate_length):
    list_recordids=df.RecordID.unique()
    list_parametrs=df.Parameter.unique()
    excluded_parameters=['RecordID','Age','Gender','Height','ICUType']
    list_parametrs=np.setdiff1d(list_parametrs, excluded_parameters)
    
    print(list_parametrs)

    VALUES,TIMESTAMPS,MASKS,PADDINGS,OUTCOMES,OUTCOMES_2,PATIENT_ID=[],[],[],[],[],[],[]
    for recordid in tqdm(list_recordids):
        sub_df=df.loc[df.RecordID==recordid,:].sort_values(by='Time') 
        tempo_values,tempo_timestamps,tempo_masks,tempo_paddings=[],[],[],[]
        for parameter in list_parametrs:
            sub_sub_df=sub_df.loc[sub_df.Parameter==parameter,:]
            values=sub_sub_df.Value.values

            timestamps=sub_sub_df.Time.values
            masks=sub_sub_df.Mask.values
            timestamps=np.round(np.array(list(map(lambda x: int(x.split(':')[0])+(int(x.split(':')[1])/60),timestamps))),2)
            
            if len(values)==0:
                values=np.array([0])
                timestamps=np.array([0])
                masks=np.array([0])
                paddings=np.array([0])
            else:
                paddings=np.ones_like(values)

            masks=np.pad(masks[-univariate_length:],(0,univariate_length-len(masks[-univariate_length:])),'edge')
            paddings=np.pad(paddings[-univariate_length:],(0,univariate_length-len(paddings[-univariate_length:])),'constant',constant_values=0)
            values=np.pad(values[-univariate_length:],(0,univariate_length-len(values[-univariate_length:])),'edge')
            timestamps=np.pad(timestamps[-univariate_length:],(0,univariate_length-len(timestamps[-univariate_length:])),'edge')
            
            
            
            tempo_paddings.append(paddings)
            tempo_values.append(values)
            tempo_timestamps.append(timestamps)
            tempo_masks.append(masks)
            
    
        outcome=outcomes.loc[outcomes.RecordID==recordid,:]['In-hospital_death'].values
        outcome_2=outcomes.loc[outcomes.RecordID==recordid,:]['SOFA'].values
        
        
        OUTCOMES.append(outcome)
        OUTCOMES_2.append(outcome_2)
        PATIENT_ID.append(recordid)
        VALUES.append(tempo_values)
        TIMESTAMPS.append(tempo_timestamps)
        MASKS.append(tempo_masks) 
        PADDINGS.append(tempo_paddings) 
    
    return np.array(VALUES),np.array(TIMESTAMPS),np.array(MASKS),np.array(PADDINGS),np.array(OUTCOMES),np.array(OUTCOMES_2),np.array(PATIENT_ID)

def timeVariationBuilding(timestamp_matrix,mask_matrix):
    Delta_time=[]
    for k in range(timestamp_matrix.shape[0]):
        all_=[]
        for j in range(timestamp_matrix.shape[1]):
            tempo=[]
            for p in range(timestamp_matrix.shape[2]):
                if j==0:
                    # at t=1 in the paper
                    tempo.append(0)
                else:
                    #If m^k_(t-1)=0 (in the paper)
                    if mask_matrix[k][j-1][p]==0:
                        tempo.append(timestamp_matrix[k][j][p]-timestamp_matrix[k][j-1][p]+all_[j-1][p])
                    else:
                        tempo.append(timestamp_matrix[k][j][p]-timestamp_matrix[k][j-1][p])
            all_.append(tempo)
        Delta_time.append(all_)
    Delta_time=np.array(Delta_time) 
    return Delta_time

def maskImputation(percentage,original_mask,option='train'):
    imputer_mask=copy.deepcopy(original_mask)
    distribution_mask=Counter(original_mask.reshape(-1))
    print(f'Total number of true observations {distribution_mask[1.]}')
    percentage_drop=(distribution_mask[1.]*percentage)//100
    print(f'Number of observed values used for imputation/padding {percentage_drop}')
    
    
    
    imputer_mask=imputer_mask.reshape(-1)
    if option=='test':
        MASKTEMPO=imputer_mask
    else:
        MASKTEMPO=np.ones_like(imputer_mask)
    indexes=np.where(imputer_mask==1)
    indexes_drop=np.random.choice(indexes[0], size=percentage_drop)
    MASKTEMPO[indexes_drop]=0
    
    return MASKTEMPO.reshape(original_mask.shape[0],original_mask.shape[1],original_mask.shape[2])
            