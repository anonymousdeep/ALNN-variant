import numpy as np
import keras
from tensorflow.keras import layers
import tensorflow as tf


class ALNNLayer(tf.keras.layers.Layer):

    def __init__(self,init_time=0,prior_hours=48,time_space=13,type_distance="abs",alnn_dropout=0.0):
        super(ALNNLayer, self).__init__()
        self.prior_hours = prior_hours
        self.init_time = init_time
        self.time_space=time_space
        self.nr_ref_time_points=time_space
        self.type_distance=type_distance
        self.alnn_dropout=alnn_dropout


#         if((self.prior_hours%self.time_space)!=0):
#             raise Exception(f'{self.time_space}  must be a multiple of {self.prior_hours}.')

        #Reference time points
        self.ref_time=np.linspace(init_time,self.prior_hours,self.nr_ref_time_points)
        self.ref_time=self.ref_time.reshape(self.nr_ref_time_points,1,1)

        self.dropout_1=layers.Dropout(self.alnn_dropout)
        self.dropout_2=layers.Dropout(self.alnn_dropout)
        self.normalize=layers.Normalization()


    def build(self, input_shape):

        self.axis_2=input_shape[0][1]
        self.axis_3=input_shape[0][2]

        self.alpha = self.add_weight(shape=(self.nr_ref_time_points,1,1),
                                 initializer='glorot_uniform',
                                 name='alpha',
                                 dtype='float32',
#                                  regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.0),
                                 trainable=True)

#         self.w_v = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2]),
        self.w_v = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2]),
                                 initializer='glorot_uniform',
                                 name='w_inyensity',
                                 dtype='float32',
                                 trainable=True)

        self.w_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2],4),
#         self.w_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,1,3),
                                 initializer='glorot_uniform',
                                 name='w_tempo',
                                 dtype='float32',
                                 trainable=True)

#         self.w_l = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,input_shape[0][2],self.axis_2,input_shape[0][2]),
# #         self.w_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2,1,3),
#                                  initializer='glorot_uniform',
#                                  name='w_linear',
#                                  dtype='float32',
#                                  trainable=True)
        self.b_v= self.add_weight(shape=(self.nr_ref_time_points,1,input_shape[0][2]),
                                 initializer='glorot_uniform',
                                 name='bias_intensity',
                                 dtype='float32',
                                 trainable=True)

        self.b_t = self.add_weight(shape=(self.nr_ref_time_points,self.axis_2, self.axis_3,1),
                                 initializer='glorot_uniform',
                                 name='bias_tempo',
                                 dtype='float32',
                                 trainable=True)

    def call(self, inputs,training=None):
        self.X=inputs[0]#values
        self.T=inputs[1]#timestamps
        self.M=inputs[2]#masks
        self.PD=inputs[3]#padding mask


        #Dupliction with respect to the number of reference time points
        self.x=tf.tile(self.X[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.t=tf.tile(self.T[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.m=tf.tile(self.M[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        self.pd=tf.tile(self.PD[:,None,:,:],[1,self.nr_ref_time_points,1,1])
        
        if(self.type_distance=="abs"):
            self.diastance=tf.abs(self.t-tf.cast(self.ref_time,tf.float32))
        else:
            self.diastance=tf.square(self.t-tf.cast(self.ref_time,tf.float32))

        self.kernel=tf.exp(-tf.cast(tf.nn.relu(self.alpha),tf.float32)*self.diastance)
        #time lag intensity
        self.intensity=tf.nn.relu(self.x*self.kernel)


        self.x_s=tf.reshape(self.x,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.pd=tf.reshape(self.pd,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.intensity_s=tf.reshape(self.intensity,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])
        self.m_s=tf.reshape(self.m,[-1,self.nr_ref_time_points,self.axis_2, self.axis_3,1])

        
        if training:
            #Value-level extraction
            self.lattent_x=self.dropout_1(tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.intensity_s,self.m_s,self.pd],4)+self.b_t,4)),training=training)
            #Feature-level aggregation
            self.lattent_x=self.dropout_2(tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2)),training=training)
        else:
            #Value-level extraction
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_t*tf.concat([self.x_s,self.intensity_s,self.m_s,self.pd],4)+self.b_t,4))
            #Feature-level aggregation
            self.lattent_x=tf.nn.relu(tf.reduce_sum(self.w_v*self.lattent_x + self.b_v,2))


        return self.lattent_x #pseudo-aligned latent multivariate time series

    def get_config(self):
        config = super(ALNNLayer, self).get_config()
        config.update({"prior_hours": self.prior_hours})
        config.update({"init_time": self.init_time})
        return config
    
class transitionLearning(tf.keras.layers.Layer):
    def __init__(self):
        super(transitionLearning, self).__init__()
        self.dropout=layers.Dropout(0.8)
#         self.last_dim=last_dim
    def build(self, input_shape): 
        self.alpha= self.add_weight(shape=(input_shape[1],input_shape[3],input_shape[3]),
                                initializer='glorot_uniform',
                                name='alpha',
                                dtype='float32',trainable=True)
        
    def call(self, inputs,training=None):
        time_distance=inputs
        time_distance*=self.alpha
        time_distance=tf.nn.relu(time_distance)
        if training:
            time_distance=self.dropout(time_distance) 
        return time_distance

class weightsCombineLayer(tf.keras.layers.Layer):
    def __init__(self,rate_dropout=0.5):
        super(weightsCombineLayer, self).__init__()
        self.dropout=layers.Dropout(rate_dropout)
    def build(self, input_shape): 
        self.w_t= self.add_weight(shape=(input_shape[1][2],input_shape[1][3]),
                                initializer='glorot_uniform',
                                name='glorot_uniform',
                                dtype='float32',trainable=True)
        self.w_s= self.add_weight(shape=(input_shape[0][2],input_shape[0][3]),
                                initializer='glorot_uniform',
                                name='weights_for_scores',
                                dtype='float32',trainable=True)
        
    def call(self, inputs,training=None):
        scores=inputs[0]
        time_distance=inputs[1]
        
        output= tf.nn.relu(scores*self.w_s + time_distance*self.w_t)
        if training:
            output=self.dropout(output)
        return output
    
class LinearTransformation(tf.keras.layers.Layer):
    def __init__(self,rate_drop=0.2):
        super(LinearTransformation, self).__init__()
        self.dropout=layers.Dropout(rate_drop)
    def build(self, input_shape): 
        self.w= self.add_weight(shape=(input_shape[2],input_shape[1]),
                                initializer='glorot_uniform',
                                name='w_imputation',
                                dtype='float32',trainable=True)
        
    def call(self, inputs,training=None):
        
        output=inputs@self.w
        if training:
            output=self.dropout(output)
 
        return output
    
class ScoringLayer(tf.keras.layers.Layer):
    def __init__(self,):
        super(ScoringLayer, self).__init__()
        self.dropout=layers.Dropout(0.0)
    def build(self, input_shape):        
        self.w_q= self.add_weight(shape=(input_shape[1],input_shape[1]),
                                   initializer='glorot_uniform',name='w_querry',dtype='float32',trainable=True)
        
        self.w_k= self.add_weight(shape=(input_shape[1],input_shape[1]),
                                   initializer='glorot_uniform',name='w_key',dtype='float32',trainable=True)
          
    def call(self, inputs,training=None):
        deltas=inputs

        q=tf.transpose(deltas,[0,2,1])@self.w_q
        k=tf.transpose(deltas,[0,2,1])@self.w_k

        
        dim=tf.sqrt(tf.cast(q.shape[1],tf.float32))
        output=tf.nn.softmax(q@tf.transpose(k,[0,2,1])/dim)
        if training:
            output=self.dropout(output)
        return output

class balancer(tf.keras.layers.Layer):
    def __init__(self,):
        super(balancer, self).__init__()
        self.dropout=layers.Dropout(0.0)
    def build(self, input_shape): 
        self.theta= self.add_weight(shape=(input_shape[0][2],input_shape[0][2]),
                                initializer='glorot_uniform',
                                name='theta',
#                                 constraint=keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, axis=-1),
                                dtype='float32',trainable=True)
        
    def call(self, inputs,training=None):
        i1=inputs[0]
        i2=inputs[1]
        mask=inputs[2]
        
        balancer=tf.nn.sigmoid(mask@self.theta)
        output=balancer*i1 + (1-balancer)*i2
        if training:
            output=self.dropout(output)
        return output

class ALNN_VARIANT(keras.Model):

    def __init__(self,
                 init_time=0,
                 max_time=48,
                 time_interval=1,
                 type_distance="abs",
                 gru_unit=168,
                 gru_dropout=0.0,
                 rate_dropout_wcl=0.0,
                 pseudo_latent_dropout=0.0,
                 alnn_dropout=0.0,
                 score_loss=True,
                 n_attention=15):


        super(ALNN_VARIANT, self).__init__()

        self.max_time=max_time
        self.init_time=init_time
        self.type_distance=type_distance
        self.gru_unit=gru_unit
        self.gru_dropout=gru_dropout
        self.pseudo_latent_dropout=pseudo_latent_dropout
        self.time_interval=time_interval
        self.time_interval=self.max_time*self.time_interval+1
        self.score_loss=score_loss
        self.alnn_dropout=alnn_dropout
        self.rate_dropout_wcl=rate_dropout_wcl
        self.n_attention=n_attention
        
        self.ALNNLayer=ALNNLayer(self.init_time,
                                self.max_time,
                                self.time_interval,
                                self.type_distance,
                                self.alnn_dropout)
        self.gru=layers.GRU(self.gru_unit,dropout=self.gru_dropout)
        
        self.intra_transitionLearning=transitionLearning()
        self.linear_transformation=LinearTransformation(rate_drop=0.0)
        self.extra_transitionLearning=transitionLearning()
        
        
        self.weights_combine_layer=weightsCombineLayer(self.rate_dropout_wcl)
        self.multi_scores=[ScoringLayer() for k in range(self.n_attention)]
        self.scores_pseudo_latent_values=ScoringLayer()
        
        self.balancer=balancer()
        
        
        
        self.dense1=layers.Dense(100)
        self.drop1=layers.Dropout(0.2)
        self.classifier=layers.Dense(1,activation='sigmoid')
        
        
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()



    def call(self, inputs,training=None):
        x=tf.cast(inputs[0],tf.float32)#matrix of values
        t=tf.cast(inputs[1],tf.float32)#matrix of timestamps
        m=tf.cast(inputs[2],tf.float32)#matrix of masks
        m_imp=tf.cast(inputs[3],tf.float32)#matrix of masks used for reconstruction loss
        pd=tf.cast(inputs[4],tf.float32)# padding masks matrix
        
        
        if training:
            m_prime=m*m_imp
        else:
            m_prime=m
        
        #Intra-imputation 
        t_transpose=tf.transpose(t,[0,2,1])
        t_prime=tf.reshape(t_transpose,[-1,t_transpose.shape[1],t_transpose.shape[2],1])
        time_weights=tf.exp(-tf.abs(tf.reshape(t_transpose,[-1,t_transpose.shape[1],1,t_transpose.shape[2]])-t_prime))
        time_weights=self.intra_transitionLearning(time_weights)
        time_weights=tf.nn.softmax(time_weights,-1)
        x_transpose=tf.transpose(x*m_prime,[0,2,1])
        in_imputed_values=tf.reshape(x_transpose,[-1,x_transpose.shape[1],1,x_transpose.shape[2]])*time_weights
        in_imputed_values=tf.reduce_sum(in_imputed_values,-1)
        in_imputed_values=tf.transpose(in_imputed_values,[0,2,1])


        #Extra-imputation
        tprime=tf.exp(-tf.abs(tf.reshape(t,[-1,t.shape[1],t.shape[2],1])-tf.reshape(t,[-1,t.shape[1],1,t.shape[2]])))
        tprime=tf.transpose(tprime,[0,1,2,3])
        #scores-> mixture of similarity score matrices
        scores=tf.concat([corr(x*m_prime) for corr in self.multi_scores],2)
        scores=self.linear_transformation(scores)
        scores_=tf.reshape(scores,[-1,1,scores.shape[1],scores.shape[2]])
        scores_=self.weights_combine_layer([scores_,tprime])
        scores_=tf.nn.softmax(scores_,2)
        ex_imputed_values=tf.reshape(x*m_prime,[-1,x.shape[1],1,x.shape[2]])@scores_
        ex_imputed_values=tf.squeeze(ex_imputed_values,2)

           
        imputed_values=self.balancer([in_imputed_values,ex_imputed_values,m])
        #Imputed irregular multivariate time series (IMTS) 
        imts=m_prime*x + (1-m_prime)*imputed_values
        
        if training:
            #Reconstruction loss
            error=self.mse((1-m_imp)*x,(1-m_imp)*imputed_values)
            self.add_loss(error)
            
        lattent_data=self.ALNNLayer([imts,t,m_prime,pd])
        
        if training and self.score_loss:
            #Score loss
            error_score=self.mse(scores,self.scores_pseudo_latent_values(lattent_data))
            self.add_loss(error_score)

        #plmts->pseudo-aligned latent multivaraite time series
        plmts=self.gru(lattent_data)
       
        
        #Classifier
        plmts=self.dense1(plmts)
        plmts=self.drop1(plmts,training=training)
        plmts=self.classifier(plmts)
        
        return plmts

    def get_config(self):
        config = super(ALNN_VARIANT, self).get_config()
        config.update({"max_time": self.max_time})
        config.update({"init_time": self.init_time})
        config.update({"time_interval": self.time_interval})
        config.update({"type_of_distance": self.type_distance})
        config.update({"gru_unit": self.gru_unit})
        config.update({"gru_dropout": self.gru_dropout})
        config.update({"# of self-attention": self.n_attention})
        config.update({"pseudo_latent_dropout": self.pseudo_latent_dropout})
        return config