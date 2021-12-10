#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy.optimize


# In[11]:


import os
GPU = ""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


# In[3]:


dataset_path='../../orbit_rf_dataset/data/compact_pkl_datasets/'


# In[4]:


def load_compact_pkl_dataset(dataset_path,dataset_name):
    with open(dataset_path+dataset_name+'.pkl','rb') as f:
        dataset = pickle.load(f)
    return dataset


# In[5]:


def analyze_compact_pkl_datset(dataset):
    tx_list = dataset['tx_list']
    rx_list = dataset['rx_list'] 
    capture_date_list= dataset['capture_date_list']
    equalized_list = dataset['equalized_list'] 
    mat= np.zeros((len(equalized_list),len(capture_date_list),len(rx_list),len(tx_list))) 
    for eq_i  in range(len(equalized_list)):
        for date_i  in range(len(capture_date_list)):
            for rx_i in range(len(rx_list)):
                for tx_i in range(len(tx_list)): 
                    mat[eq_i,date_i,rx_i,tx_i] = dataset['data'][tx_i][rx_i][date_i][eq_i].shape[0]        
    return mat


# In[ ]:





# # Plot Tx-Rx grids for compact subsets

# In[6]:


dataset_name = 'ManySig'


dataset = load_compact_pkl_dataset(dataset_path,dataset_name)


mat = analyze_compact_pkl_datset(dataset)

prefix = None

for eq_i  in range(len(dataset['equalized_list'])):
        for date_i  in range(len(dataset['capture_date_list'])):
            plt.figure()
            plt.matshow(np.squeeze(mat[eq_i,date_i]))
            plt.title('Eq {} Date {}'.format(dataset['equalized_list'][eq_i],dataset['capture_date_list'][date_i] ))
            plt.colorbar()


# In[7]:


dataset_name = 'ManyTx'


dataset = load_compact_pkl_dataset(dataset_path,dataset_name)


mat = analyze_compact_pkl_datset(dataset)

prefix = None

for eq_i  in range(len(dataset['equalized_list'])):
        for date_i  in range(len(dataset['capture_date_list'])):
            plt.figure()
            plt.matshow(np.squeeze(mat[eq_i,date_i]))
            plt.title('Eq {} Date {}'.format(dataset['equalized_list'][eq_i],dataset['capture_date_list'][date_i] ))
            plt.colorbar()


# In[8]:


dataset_name = 'ManyRx'


dataset = load_compact_pkl_dataset(dataset_path,dataset_name)


mat = analyze_compact_pkl_datset(dataset)

prefix = None

for eq_i  in range(len(dataset['equalized_list'])):
        for date_i  in range(len(dataset['capture_date_list'])):
            plt.figure()
            plt.matshow(np.squeeze(mat[eq_i,date_i]))
            plt.title('Eq {} Date {}'.format(dataset['equalized_list'][eq_i],dataset['capture_date_list'][date_i] ))
            plt.colorbar()


# In[9]:


dataset_name = 'SingleDay'


dataset = load_compact_pkl_dataset(dataset_path,dataset_name)


mat = analyze_compact_pkl_datset(dataset)

prefix = None

for eq_i  in range(len(dataset['equalized_list'])):
        for date_i  in range(len(dataset['capture_date_list'])):
            plt.figure()
            plt.matshow(np.squeeze(mat[eq_i,date_i]))
            plt.title('Eq {} Date {}'.format(dataset['equalized_list'][eq_i],dataset['capture_date_list'][date_i] ))
            plt.colorbar()


# In[ ]:




