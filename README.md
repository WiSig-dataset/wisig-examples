## Overview

Example applications of the [WiSig dataset](https://cores.ee.ucla.edu/downloads/datasets/wisig).

## Citation
If you use this code in your research please cite
> S. Hanna, S. Karunaratne, and D. Cabric, “WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting,” arXiv:2112.15363 [eess], Dec. 2021, Accessed: Jan. 03, 2022. [Online]. Available: http://arxiv.org/abs/2112.15363


## Directory Description

###  Jupyter notebooks

`d001_plot_Full_WiSig_tx_rx_grid.ipynb` : plots the number of signals per Tx-Rx pairs using the file data_summary.pkl. Shows how to display the model of Tx and Rx hardware.



`d002_analyze_compact_datasets.ipynb`: Loads the compact datasets from disk and shows the number of signals per Tx-Rx pairs


`d003_ManyRx_nrx.ipynb`: Studies the impact of changing receivers on classification accuracy using the non-equalized dataset

`d011_ManyRx_nrx_eq.ipynb`: Studies the impact of changing receivers on classification accuracy using the equalized dataset


`d004_ManySig_nsig.ipynb`: Studies the impact of changing the number of training signals on classification accuracy using the non-equalized dataset


`d005_ManySig_nsig_eq.ipynb`: Studies the impact of changing the number of training signals on classification accuracy using the non-equalized dataset


`d006_ManyTx_ntx.ipynb`: Studies the impact of changing the number of Tx on classification accuracy using the non-equalized dataset


`d007_ManySig_ndays.ipynb`: Studies the impact of changing the number of training days using the non-equalized dataset


`d008_ManySig_ndays_eq.ipynb`: Studies the impact of changing the number of training days using the equalized dataset


`d009_ManyTx_localization.ipynb`:  Plots the average power received at different Rx localization


`d010_ManyTx_localization_network.ipynb`: Evaluates the performance of WiSig for localization




###  Python Files

`data_utilities.py`: Functions to load the dataset and prepare it for classification



### PKL files

`data_summary.pkl`: Contains number of signal per Tx-Rx for the entire datset


`IdSig_info.pkl`: Contains the google drive links of all files of Full WiSig


`orbit_hardware.pkl`: Contains a description of the model of WiFi Tx and USRP rx as described in Orbit


### Folders

`html`: Contain an html copy of all ipynb files


`py`: Contain a python copy of all ipynb files


`weights`: Contains the trained neural network weights
