#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
from numpy import linspace, max, min, average, std, sum, sqrt, where, argmax, mean
import matplotlib.pyplot as plt
import os
import biosignalsnotebooks as bsnb
import scipy as sp
from scipy import signal
from scipy.fft import fft, fftfreq


# In[42]:


def normalize(series):
    result = np.array(series)
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result
def filter_50Hz(data, fs=1000, quality_factor=30.0):
    
    f0 = 50.0  # Frequency to be removed from signal (Hz)

    Q = 30.0  # Quality factor

    # Design notch filter

    b_notch, a_notch = signal.iirnotch(f0, Q, fs)

    freq, h = signal.freqz(b_notch, a_notch, fs)
    outputSignal = signal.filtfilt(b_notch, a_notch, data)
    return outputSignal
    
def fft_transform(data,fs=1000,):
    T = 1.0 / fs
    N = len(data)
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(data)
    xf = fftfreq(N, T)[:N//2]
    fig = plt.figure(figsize=(20, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xlabel('Frequency (Hz)', fontsize=20)
    plt.ylabel('FFT unit', fontsize=18)
    return xf, 2.0/N * np.abs(yf[0:N//2])
    
def clean_signal(data,fs=1000, smooth_level=1, threshold_level=1):
    filter_50Hz(data)
    burst_begin, burst_end = bsnb.detect_emg_activations(data, fs, smooth_level, threshold_level, 
                                                     time_units=True, plot_result=True)[:2]
    burst_begin, burst_end = (burst_begin*1000).astype(int), (burst_end*1000).astype(int)
    
    data[0:burst_begin[0]]=0
    for i in range(0,len(burst_begin)-1):
        data[burst_end[i]:burst_begin[i+1]] = 0
        


# In[27]:


path = os.path.join(os.path.realpath(''),'EMG Physical Action Data Set')
savepath_filteredsignal=os.path.join(os.path.realpath(''),'Filtered_Signal')


# In[41]:


emg_clapping_Rbicep = []
time = []
fs = 1000
for folder in os.listdir(path):
  if folder != "readme.txt":
    subpath = path + "/" + folder + "/Normal/txt/Clapping.txt"
    data = normalize(np.loadtxt(subpath)[:, 0])
    time.append(linspace(0, len(data) / fs, len(data)))
    emg_clapping_Rbicep.append(clean_signal(data))    
    #np.savetxt(savepath_filteredsignal+'/'+folder+'_Clapping'+'.txt',data)
    #print(savepath_filteredsignal+'/'+folder+'_Clapping'+'.txt')


# In[30]:


from scipy import signal

import matplotlib.pyplot as plt

fs = 1000 # Sample frequency (Hz)

f0 = 50.0  # Frequency to be removed from signal (Hz)

Q = 30.0  # Quality factor

# Design notch filter

b_notch, a_notch = signal.iirnotch(f0, Q, fs)

freq, h = signal.freqz(b_notch, a_notch, fs)

fig = plt.figure(figsize=(8, 6))
  
# Plot magnitude response of the filter
plt.plot(freq*fs/(2*np.pi), 20 * np.log10(abs(h)),
         'r', label='Bandpass filter', linewidth='2')
  
plt.xlabel('Frequency [Hz]', fontsize=20)
plt.ylabel('Magnitude [dB]', fontsize=20)
plt.title('Notch Filter', fontsize=20)
plt.grid()

