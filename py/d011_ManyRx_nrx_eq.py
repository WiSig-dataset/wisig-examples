#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import os.path

import scipy,scipy.spatial
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

from  data_utilities import *
# from definitions import *
# from run_train_eval_net import run_train_eval_net,run_eval_net


# In[2]:


import os
GPU = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


# In[3]:


dataset_name = 'ManyRx'
dataset_path='../../orbit_rf_dataset/data/compact_pkl_datasets/'

compact_dataset = load_compact_pkl_dataset(dataset_path,dataset_name)

tx_list = compact_dataset['tx_list']
rx_list = compact_dataset['rx_list']

equalized = 1

capture_date_list = compact_dataset['capture_date_list']
capture_date = capture_date_list[0]
n_tx = len(tx_list)
n_rx = len(rx_list)
print(n_tx,n_rx)


# In[4]:


np.random.seed(0)
n_real = 5
rx_list_real = []
for i in range(n_real):
    np.random.shuffle(rx_list)
    rx_list_real.append(np.copy(rx_list).tolist())
print(rx_list_real)


# In[5]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
 


# In[6]:


def create_net():


   inputs = Input(shape=(256,2))
   x = Reshape((256,2,1))(inputs)
   x = Conv2D(8,(3,2),activation='relu',padding = 'same')(x)
   x = MaxPool2D((2,1))(x)
   x = Conv2D(16,(3,2),activation='relu',padding = 'same')(x)
   x = MaxPool2D((2,1))(x)
   x = Conv2D(16,(3,2),activation='relu',padding = 'same')(x)
   x = MaxPool2D((2,2))(x)
   x = Conv2D(32,(3,1),activation='relu',padding = 'same')(x)
   x = MaxPool2D((2,1))(x)
   x = Conv2D(16,(3,1),activation='relu',padding = 'same')(x)
   #x = resnet(x,64,(3,2),'6')
   #x = MaxPool2D((2,2))(x)
   x = Flatten()(x)



   x = Dense(100, activation='relu', kernel_regularizer = keras.regularizers.l2(0.0001))(x)
   # x = Dropout(0.3)(x)
   x = Dense(80, activation='relu',kernel_regularizer = keras.regularizers.l2(0.0001))(x)
   x = Dropout(0.5)(x)
   x = Dense(n_tx, activation='softmax',kernel_regularizer = keras.regularizers.l2(0.0001))(x)
   ops = x

   classifier = Model(inputs,ops)
   classifier.compile(loss='categorical_crossentropy',metrics=['categorical_accuracy'],optimizer=keras.optimizers.Adam(0.0005))
   
   return classifier

classifier = create_net()
classifier.summary()


# In[7]:


def evaluate_test(classifier):
    pred = classifier.predict(sig_dfTest)
    acc = np.mean(np.argmax(pred,1)==txidNum_dfTest)

    test_indx = ()
    for indx in range(len(tx_list)):
        cls_indx = np.where(txidNum_dfTest == indx)
        test_indx = test_indx + (cls_indx[0][:n_test_samples],)
    test_indx = np.concatenate(test_indx) 
    acc_bal = np.mean(np.argmax(pred[test_indx,:],1)==txidNum_dfTest[test_indx])
    return acc,acc_bal


# In[8]:


n_test_rx = 5;


# In[9]:



list(range( 0,len(rx_list_real[0])-n_test_rx+1,5)) 


# In[10]:


TRAIN = True
continue_training = True
nreal = 5

real_list = list(range(nreal))
nrx_list =  list(range( 0,len(rx_list_real[0])-n_test_rx+1,5)) 

patience = 5
n_epochs = 100

smTest_results = []
dfTest_results = []
dfTestBal_results = []

for real in real_list:
    rx_list = rx_list_real[real]
    rx_test_list = rx_list[-n_test_rx:]
    test_dataset =  merge_compact_dataset(compact_dataset,capture_date,tx_list,rx_test_list,equalized=equalized)
    test_augset_dfRx,_,_ = prepare_dataset(test_dataset,tx_list,val_frac=0.0, test_frac=0.0)

    [sig_dfTest,txidNum_dfTest,txid_dfTest,cls_weights] = test_augset_dfRx

    cnt=np.histogram(txidNum_dfTest,bins=np.arange(len(tx_list)+1)-0.5)
    n_test_samples = int(np.min(cnt[0]))

    smTest_results_real = []
    dfTest_results_real = []
    dfTestBal_results_real = []
    for nrx in nrx_list:
        print("");print("")
        print("nrx: {} - real: {} ".format(nrx,real))
        fname_w = 'weights/d011_{:02d}_{:02d}.hd5'.format(nrx,real)
        rx_train_list= rx_list[:nrx+1]

        dataset =  merge_compact_dataset(compact_dataset,capture_date,tx_list,rx_train_list,equalized=equalized)

        train_augset,val_augset,test_augset_smRx =  prepare_dataset(dataset,tx_list,
                                                            val_frac=0.1, test_frac=0.1)
        [sig_train,txidNum_train,txid_train,cls_weights] = train_augset
        [sig_valid,txidNum_valid,txid_valid,_] = val_augset
        [sig_smTest,txidNum_smTest,txid_smTest,cls_weights] = test_augset_smRx
        
        if continue_training:
            skip = os.path.isfile(fname_w) or os.path.isfile(fname_w+'.index')
        else:
            skip = False
        classifier = create_net()
        if TRAIN and not skip:
            filepath = 't_weights_'+GPU
            c=[ keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True),
              keras.callbacks.EarlyStopping(monitor='val_loss',  patience=patience)]
            history = classifier.fit(sig_train,txid_train,class_weight=cls_weights,
                                     validation_data=(sig_valid , txid_valid),callbacks=c, epochs=n_epochs)
            classifier.load_weights(filepath)
            classifier.save_weights(fname_w)
        else:
            classifier.load_weights(fname_w).expect_partial()

        smTest_r = classifier.evaluate(sig_smTest,txid_smTest,verbose=0)[1]
    #     dfTest_r = classifier.evaluate(sig_dfTest,txid_dfTest)[1]
        dfTest_r,dfTestBal_r = evaluate_test(classifier)

        print(smTest_r,dfTest_r)
        smTest_results_real.append(smTest_r)
        dfTest_results_real.append(dfTest_r)
        dfTestBal_results_real.append(dfTestBal_r)
        K.clear_session()
    smTest_results.append(smTest_results_real)
    dfTest_results.append(dfTest_results_real)
    dfTestBal_results.append(dfTestBal_results_real)    
    
    
    


# In[11]:


nrx_list


# In[18]:


matplotlib.rcParams['figure.dpi'] = 100
plt.errorbar(np.array(nrx_list)+1,np.mean(smTest_results,0),np.std(smTest_results,0),capsize=4)
plt.errorbar(np.array(nrx_list)+1,np.mean(dfTest_results,0),np.std(dfTest_results,0),capsize=4)
plt.legend(['Same Rx(s)','Diff. Rx'])
plt.xlabel('N Train Rx')
plt.ylabel('Class. Accuracy')
#plt.xticks(range(0,len(nrx_list),2))
plt.grid()
print(np.mean(dfTest_results,0).tolist())


# In[13]:


print(tx_list)
print(nrx_list)
print(real_list)
print(smTest_results)
print(dfTest_results)
print(dfTestBal_results)


# In[14]:


print(rx_list_real)


# In[ ]:




