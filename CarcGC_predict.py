import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History,CSVLogger,ReduceLROnPlateau
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from Car_model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse
from sklearn.model_selection import KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

####################################Settings#################################
parser = argparse.ArgumentParser(description='Chemical_Genotoxicity_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, default=True, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()
random.seed(0)
tf.set_random_seed(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(0)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
israndom=args.israndom
GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = GCN_deploy

####################################Constants Settings###########################
Drug_feature_file = r'CarcGC_data\drug_graph_feat'
Max_atoms = 100

def DataGenerate(Drug_feature_file):
    # load drug features
    drug_pubchem_id_set = []
    all_drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        all_drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(all_drug_feature.values())
    return all_drug_feature

def MetadataGenerate(all_drug_feature):
    label = pd.read_csv(r'Dataset\train.csv',index_col = None, header = 0)
    drugnames = label['pert_id'].tolist()
    label['Carcinogenicity_label'] = (label['Carcinogenicity'] == '+').astype(int)
    data_idx = list(zip(label['pert_id'],label['Carcinogenicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    return drug_feature,data_idx

def ValidationGenerate(all_drug_feature):
    label = pd.read_csv(r'Dataset\validation.csv',index_col = None, header = 0)
    drugnames = label['pert_id'].tolist()
    label['Carcinogenicity_label'] = (label['Carcinogenicity'] == '+').astype(int)
    data_idx = list(zip(label['pert_id'],label['Carcinogenicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    return drug_feature,data_idx

def ExtraGenerate(all_drug_feature):
    label = pd.read_csv(r'Dataset\test.csv',index_col = None, header = 0)
    drugnames = label['pert_id'].tolist()
    label['Carcinogenicity_label'] = (label['Carcinogenicity'] == '+').astype(int)
    data_idx = list(zip(label['pert_id'],label['Carcinogenicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    return drug_feature,data_idx

def AirDataGenerate():
    drug_pubchem_id_set = []
    all_drug_feature = {}
    for each in os.listdir(r'CarcGC_data\drug_graph_feat_air'):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(r'CarcGC_data\drug_graph_feat_air',each))
        all_drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(all_drug_feature.values())
    label = pd.read_csv(r'Dataset\haps.csv',index_col = None, header = 0)
    drugnames = label['CASRN'].tolist()
    label['Carcinogenicity_label'] = (label['Carcinogenicity'] == '+').astype(int)
    data_idx = list(zip(label['CASRN'],label['Carcinogenicity_label']))
    nb_drugs = len(set([item[0] for item in data_idx]))
    drug_feature = {key: all_drug_feature[key] for key in drugnames} 
    return drug_feature,data_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def FeatureExtract(data_idx,drug_feature):
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance,dtype='int16')
    for idx in range(nb_instance):
        drugname,clabel = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(drugname)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        target[idx] = clabel
    # return drug_data,gexpr_data,toxicity
    return drug_data,target

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    return 2.0*prec*recal/(prec+recal+K.epsilon())

def average_precision(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.double) 

class MyCallback(Callback):
    def __init__(self,training_data,validation_data,patience):
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.losses = {'batch':[], 'epoch':[]}
        self.auct = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.aucl = {'batch':[], 'epoch':[]}
        self.H = {}
        return
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('CarcGC_data/bestmodel/BestCarcGC_Cartoxicity_classify_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        y_pred_train = self.model.predict(self.x_train)
        roc_train = roc_auc_score(self.y_train, y_pred_train)
        precision,recall,_, = metrics.precision_recall_curve(self.y_val,y_pred_val)
        pr_val = -np.trapz(precision,recall)
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.auct['epoch'].append(roc_train)
        self.aucl['epoch'].append(roc_val)
        print('roc-val: %.4f, pr-val:%.4f' % (roc_val,pr_val))
        if roc_val > self.best:
            self.best = roc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
            self.model.save('CarcGC_data/bestmodel/BestCarcGC_Cartoxicity_highestAUCROC_%s.h5'%model_suffix)
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return
    def savedata(self,lr,batchsize,wd,dd):
        iters = range(len(self.val_loss['epoch']))
        eponb = float(len(self.losses['batch']))/float(len(self.val_loss['epoch']))
        dflist = []
        for ii in iters:
            ystart = int(ii*eponb)
            yend = int((ii+1)*eponb)
            yloss = self.losses['epoch'][ii]
            valloss = self.val_loss['epoch'][ii]
            aucroct = self.auct['epoch'][ii]
            aucroc = self.aucl['epoch'][ii]
            dflist.append([ii+1,aucroct,aucroc,yloss,valloss])
        df = pd.DataFrame(dflist)
        df.columns = ['epoch','auc_train','auc_val','train_loss','validation_loss']
        df.to_csv('CarcGC_data/gridsearch_loss/lr%s_batch%s_dropout%s_%sl2_loss.csv'%(lr,batchsize,dd,wd),index=False,header=True)
        return

def ModelTraining(model,lr,batchsize,wd,dd,X_drug_data_train,Y_train,validation_data,nb_epoch=500):
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='binary_crossentropy',metrics=['accuracy',precision,recall,f1_score,average_precision])
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    history = MyCallback(training_data = [[X_drug_feat_data_train,X_drug_adj_data_train],Y_train],validation_data=validation_data,patience=80)
    callbacks = [ModelCheckpoint(filepath='CarcGC_data/checkpoint_weight/Cartoxicity_weights_{epoch:04d}.h5',verbose=1),history]
    model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train],validation_data=([validation_data[0][0],validation_data[0][1]], validation_data[1]),y=Y_train,batch_size=batchsize,epochs=nb_epoch,callbacks=callbacks)
    history.savedata(lr,batchsize,wd,dd)
    return model

def ModelEvaluate(model,X_drug_data_test,Y_test,cancer_type_test_list):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test])
    auROC_all = metrics.roc_auc_score(Y_test, Y_pred)
    fpr,tpr,_,= metrics.roc_curve(Y_test,Y_pred)
    precision,recall,_, = metrics.precision_recall_curve(Y_test,Y_pred)
    auPR_all = -np.trapz(precision,recall)
    print("The overall AUC and auPR is %.4f and %.4f."%(auROC_all,auPR_all))
    return auROC_all,auPR_all,Y_pred

def main():
    lr, batchsize, dd, wd = 0.00005, 32, 0.1, 0.001
    print(lr,batchsize,dd,wd)
    random.seed(0)
    tf.set_random_seed(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(0)
    all_drug_feature= DataGenerate(Drug_feature_file)
    drug_feature,data_idx = MetadataGenerate(all_drug_feature)
    X_drug_data_train,Y_train = FeatureExtract(data_idx,drug_feature)
    drug_feature_validation,data_idx_validation = ValidationGenerate(all_drug_feature)
    X_drug_data_validation,Y_validation = FeatureExtract(data_idx_validation,drug_feature_validation)
    X_drug_feat_data_validation = [item[0] for item in X_drug_data_validation]
    X_drug_adj_data_validation = [item[1] for item in X_drug_data_validation]
    X_drug_feat_data_validation = np.array(X_drug_feat_data_validation)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_validation = np.array(X_drug_adj_data_validation)#nb_instance * Max_stom * Max_stom  
    validation_data = [[X_drug_feat_data_validation,X_drug_adj_data_validation],Y_validation]
    model = KerasMultiSourceGCNModel(regr=False).createMaster(X_drug_data_train[0][0].shape[-1],args.unit_list,wd,dd,args.use_relu,args.use_bn,args.use_GMP)
    print('Begin training...')
    model = ModelTraining(model,lr,batchsize,wd,dd,X_drug_data_train,Y_train,validation_data,nb_epoch=500)
    auROC_train,auPR_train,Y_pred_train = ModelEvaluate(model,X_drug_data_train,Y_train,r'CarcGC_data\CarcGC_%s.log'%(model_suffix))
    auROC_validation,auPR_validation,Y_pred_validation = ModelEvaluate(model,X_drug_data_validation,Y_validation,r'CarcGC_data\CarcGC_test_%s.log'%(model_suffix))
    valres = pd.DataFrame([Y_validation,Y_pred_validation])
    valres.to_csv('CarcGC_data/predict_result/lr%s_batch%s_dropout%s_%sl2_validation_res.csv'%(lr,batchsize,dd,wd),index=False,header=False)
    drug_test_feature,data_test_idx = ExtraGenerate(all_drug_feature)
    X_drug_data_test,Y_test = FeatureExtract(data_test_idx,drug_test_feature)
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    auROC_test,auPR_test,Y_pred_test = ModelEvaluate(model,X_drug_data_test,Y_test,r'CarcGC_data\CarcGC_test_%s.log'%(model_suffix))
    testres = pd.DataFrame([Y_test,Y_pred_test])
    testres.to_csv('CarcGC_data/predict_result/lr%s_batch%s_dropout%s_%sl2_extra_res.csv'%(lr,batchsize,dd,wd),index=False,header=False)

    drug_feature_air,data_air_idx = AirDataGenerate()
    X_drug_data_air,Y_air = FeatureExtract(data_air_idx,drug_feature_air)
    X_drug_feat_data_air = [item[0] for item in X_drug_data_air]
    X_drug_adj_data_air = [item[1] for item in X_drug_data_air]
    X_drug_feat_data_air = np.array(X_drug_feat_data_air)
    X_drug_adj_data_air = np.array(X_drug_adj_data_air)
    Y_pred_air = model.predict([X_drug_feat_data_air,X_drug_adj_data_air])
    airlist = []
    for i,scolist in enumerate(Y_pred_air):
        sco = scolist[0]
        chem1 = data_air_idx[i]
        thre1 = 0.59459144
        if sco>=thre1:
            airlist.append([chem1,1])
        else:
            airlist.append([chem1,0])
    airres = pd.DataFrame(airlist)
    airres.to_csv('CarcGC_data/predict_result/haps_result.csv',index=False,header=False)

if __name__=='__main__':
    main()