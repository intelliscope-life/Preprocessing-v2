# Title:                ECG_Scaleograms.py
# Description:          Generates scaleograms of ECG signals and save them as RGB images
# Authors:              Kithmini Herath
# Created:              2020-10-04

# Inputs:
# Denoised ECG signals

# Outputs:
# Images of scaleogram plots of every ECG signal

# Libraries:
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Used to remove unnecessary warning during runtime
import wfdb
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from wfdb import processing
import pywt
# Library implemented by Alsauve available at: https://github.com/alsauve/scaleogram
# This may or may not be the version edited by me
import scaleogram as scg                   

# Path Specification:
data2017path = '/home/ec2-user/SageMaker/ECG_analysis/Denoised2017/'    # Data Set path
figpath = '/home/ec2-user/SageMaker/ECG_analysis/denoised_scalograms/'  # Image output path

def readData(filename):
    ''' Reading .mat ecg files 
    
    Input: File name as a string. Example file name: 'D_A00001.mat'
    Output: Numpy array containing samples of the ECG signal
    
    '''
    
    fname = filename[:-4]
    data = scipy.io.loadmat(data2017path + filename)
    ecg = data['y'].T
    
    return ecg

def qrs_detection(ecg,fs,plot):
    ''' Learned QRS Complex Detection
    
    Inputs: ecg - 1D numpy array containing ECG samples (make sure a ND numpy array is flattened)
            fs - Sampling frequency
            plot - True/False indicating if plotting detections is required
    Output: Indices of the R peaks in the numpy array that contains the ECG samples
    
    '''

    qrs_indices = processing.xqrs_detect(sig=ecg, fs=fs,learn = True, verbose = False)
    
    # If plotting the detected QRS complexes is required:
    if plot:
        wfdb.plot_items(signal=ecg, ann_samp=[qrs_indices],figsize=(20,10))

    return qrs_indices

fs = 300.0 # Sampling Frequency of ECG signals

# Brief scale to frequency mapping explanation:
# dt = 1/300  # when 300 Hz is the sampling frequency
# How the 'scale' parameter is mapped to frequency when using complex morlet wavelet
# scales = [1, 2, 3, 4]
# frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
# frequencies = [300., 150., 100.,  75.]

# Obtaining Continuous Wavelet Transform (CWT) and plotting using scaleogram library:
scg.set_default_wavelet('cmor1.5-1.0')            # Default wavelet to be used must be specified.
# Complex Morlet Wavelet used here with bandwidth parameter = 1.5 and center frequency = 1.0 Hz

# Useful bandwidth of ECG = (0.5,100)Hz
# The higher the scale applied to the transform, the wavelet becomes more sensitive to low frequencies of the given signal (this can be observed in the above scale to frequency mapping example explained above)
# Increased scale thus contributes to the smoothening of the X axis
# Reference: https://github.com/alsauve/scaleogram/blob/master/doc/scale-to-frequency.ipynb
scales = np.arange(2, 450)                        # Range for the scale must be specified for the wavelet

def plot_wavelet(ecg,fs,scales,xlim,title):
    ''' Computing CWT and Plotting Scalogram 
    
    Inputs: ecg - 1D numpy array containing ECG samples
            fs - Sampling frequency (of float data type)
            scales - The range of scales to be used for the wavelet to be applied
            xlim - The time interval to be used for the scaleogram plot
            title - Figure name used when saving the plot as a figure (figpath+filename)
    Output: An image of the scaleogram plot
    
    '''
    
    t = np.array(range(len(ecg)-1))/fs            # Timestamps of the ecg samples
    cwt = scg.CWT(t, ecg,scales)                  # Defining the CWT instance
    scg.cws(cwt, figsize = (10,5),xlim = xlim, yaxis='frequency',yscale = 'log',title = title,cscale='linear',cbar=None) # Applying  the wavelet transform and plotting and saving the scaleogram as an image


labels_csv = pd.read_csv('/home/ec2-user/SageMaker/ECG_analysis/physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv',names = ['File_name','Class'],index_col = 'File_name')   
labels = np.array(labels_csv['Class'])            # Labels of the ECG signals in the Dataset
# PhysioNet 2017 labels: 'Normal', 'AF', 'Other', 'Noise' ('Other' label corresponds to heart diseases excluding AF)

# For 4 peaks the number of samples to take from the start was chosen as 1000, since the number of examples in the dataset where sample_dif (the number of samples that captures 4 peaks) > 1000 was around 552
# For 6 peaks the number of samples to take from the start was chosen as 1700, since the number of examples in the dataset where sample_dif > 1700 was around 451

matfiles = sorted(os.listdir(data2017path))       # Storing all ECG .mat filenames
files = [x for x in matfiles if x[-3:] == 'mat']  # Making sure all loaded files are .mat files
count = 1                                         # Counting the number of files processed
for file in files:
    if count%500 == 0:
        print('%d files processed...'%count)
    data_ = readData(file)                        # Loading the data file
    ecg = data_.flatten()
    label = list(labels_csv.loc[file[2:8]])[0]    # The class of the example
    
    # Dropping the first few samples based on the length of the ecg array and obtaining indices of detected R peaks
    if len(ecg) <= 5000:
        qrs_indices = qrs_detection(ecg[499:],fs,plot = False)       # First 499 samples dropped
    else:
        qrs_indices = qrs_detection(ecg[1499:],fs,plot = False)      # First 1499 samples dropped
        
    len_indi_l = len(qrs_indices)                 # Total number of R peaks detected in the ECG signal
    
    if len_indi_l >= 8:
        if len_indi_l%2 == 1:                     # When the number of R peaks detected is an odd number
            Rpeaks = qrs_indices[round(len_indi_l/2)-1:round(len_indi_l/2)+3]   # Getting 4 R peaks from the center of the ECG      signal
            plot_wavelet(ecg,fs,scales,(Rpeaks[0]/300.0,(Rpeaks[0]+1000)/300.0),figpath+file[2:8]+'_'+label) # Obtaining the scaleogram
        else:                                     # When the number of R peaks detected is an even number
            Rpeaks = qrs_indices[round(len_indi_l/2)-2:round(len_indi_l/2)+2]   # Getting 4 R peaks from the center of the ECG      signal
            plot_wavelet(ecg,fs,scales,(Rpeaks[0]/300.0,(Rpeaks[0]+1000)/300.0),figpath+file[2:8]+'_'+label) # Obtaining the scaleogram
        
    count += 1