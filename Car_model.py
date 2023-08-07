import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers.graph import GraphLayer,GraphConv

class KerasMultiSourceGCNModel(object):
    def __init__(self,regr=False):#
        self.regr = regr
    def createMaster(self,drug_dim,units_list,wd,dd,use_relu=True,use_bn=True,use_GMP=True):
        ddseed = 1
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        #drug feature with GCN
        GCN_layer = GraphConv(units=units_list[0],step_num=1,kernel_regularizer=l2(wd))([drug_feat_input,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        for i in range(len(units_list)-2):
            GCN_layer = GraphConv(units=units_list[i+1],step_num=1,kernel_regularizer=l2(wd))([GCN_layer,drug_adj_input])
            if use_relu:
                GCN_layer = Activation('relu')(GCN_layer)
            else:
                GCN_layer = Activation('tanh')(GCN_layer)
            if use_bn:
                GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = GraphConv(units=100,step_num=1,kernel_regularizer=l2(wd))([GCN_layer,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)
        x = x_drug
        x = Dense(100,activation = 'tanh')(x)
        x = Dropout(dd,seed = ddseed)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=10, kernel_size=(1,50),strides=(1, 1), activation = 'relu',padding='valid',kernel_regularizer=l2(wd))(x) #
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid',kernel_regularizer=l2(wd))(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid',kernel_regularizer=l2(wd))(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Flatten()(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input],outputs=output)  
        return model