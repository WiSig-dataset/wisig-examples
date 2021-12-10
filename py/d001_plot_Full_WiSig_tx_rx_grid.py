#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


# In[2]:


import os
GPU = ""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


# In[3]:


with open('data_summary.pkl', 'rb') as f:
    d=pickle.load(f)
capture_date_list=d['capture_date_list']
tx_list= d['tx_list']
rx_list= d['rx_list']
mat_date=np.array(d['mat_date'])
mat_date_eq=np.array(d['mat_date_eq'])


# In[38]:


from matplotlib.colors import LogNorm
def plot_mat(mat, title):
    
    
    im = plt.matshow(np.maximum(mat,1e-10),vmax=2000,origin='lower',norm=LogNorm(vmin=1, vmax=2000))
    plt.title(title)
    fig=plt.gcf()
    plt.gca().xaxis.set_ticks_position('bottom')
    t = [1, 10, 50, 250,2000]
    fig.colorbar(im,  ticks=t, format='$%.0f$')
    # plt.colorbar()
    plt.xlabel('Tx identifier')
    plt.ylabel('Rx identifier')


# # Plot Tx-Rx grid for all days (non-equalized)

# In[39]:


for mat,capture_date in zip(mat_date,capture_date_list):
    plot_mat(mat,capture_date)
    


# # Plot Tx-Rx grid for all days (equalized)

# In[40]:


for mat,capture_date in zip(mat_date_eq,capture_date_list):
    plot_mat(mat,capture_date+' eq')
    


# In[ ]:





# # View hardware specs

# In[3]:


import pickle
with open('orbit_hardware.pkl', 'rb') as f:
    [tx_hardware,rx_hardware]=pickle.load(f)
print(tx_hardware['1-1'])
print(rx_hardware['2-1'])


# In[ ]:




