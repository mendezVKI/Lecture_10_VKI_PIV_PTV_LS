# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:39:00 2021

@author: admin
"""


# We create a stochastic signal x_n and filter it with y_n.
# Then we add some noise and we study the cross coherence between input and 
# Output

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import shutil 
Folder_OUT='C:\\Users\\Admin\\OneDrive - vki.ac.be\\Mendez_Contribution\\Chap10\\Figures\\'

# Setting for the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)



# As input signal we take P2.

data = np.load('Sampled_PROBES.npz')
n_T=3000; Fs=3000; dt=1/Fs
P2_U=data['P2_U'][0:n_T];  
P3_U=data['P3_U'][0:n_T]; 

# Prepare the time axis
t=np.linspace(0,dt*(n_T-1),n_T) # prepare the time axis# 

# Define impulse response and 'input'
h=np.ones(3)/3; x=P2_U
# We padd to use the frequency domain:
h=np.concatenate([h,np.zeros(len(x)-3)]) 
# The transfer function will be
H=np.fft.fft(h)
# This is circular convolution
X=np.fft.fft(x)
# Add noise
N=np.random.normal(0, 0.5*np.std(x), len(x))
y=np.real(np.fft.ifft(X*H))+N


# Frequencies linked to this padding are:
f_fft=np.fft.fftfreq(len(H),d=1/Fs)


#  This would be with a linear convolution
# y=signal.convolve(x, h, mode='same', method='auto')+N

#plt.plot(x); plt.plot(y)

N_P=300
f, Sxx_den = signal.welch(x, Fs, nperseg=N_P,scaling='density')
f, Syy_den = signal.welch(y, Fs, nperseg=N_P,scaling='density')

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(f, 20*np.log10(Sxx_den),label='$|S_{xx}(f)|$')
plt.plot(f, 20*np.log10(Syy_den),label='$|S_{yy}(f)|$')
plt.xlim([100,1500])
plt.xlabel('$f [Hz]$',fontsize=16)
plt.ylabel('$|S_{xx}|, |S_{yy}| [db]$',fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='PSDs.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


# Prepare the transfer function of h
# curiosity: this is a nice tool for transf functions
wz, Hz = signal.freqz(h)
# plt.plot(f_fft, 20 * np.log10(abs(H)), 'r',label='|H(f)|')

plt.close('all')
# Plot The coherency
f, Cxy = signal.coherence(x, y, Fs, nperseg=N_P)
fig, ax = plt.subplots(figsize=(6, 3)) 
plt.plot(f, 20*np.log10(Cxy),label='$C_{x,y}$')
wz, Hz = signal.freqz(h)
plt.plot(wz/np.pi*Fs/2,20*np.log10(Hz), 'r',label='|H(f)|')
plt.xlim([100,1500])

plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$|C_{x,y}| [db]$',fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Cross_Coherency.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


#%% Now we proceed with the system idenfitication.
Nfft=len(x) # We use no zero padding
n_x=len(x)
# Check that all the pairs in the chapter are correct

# Zero pad:
xzp= np.concatenate([x,np.zeros(Nfft-n_x)]) # zero-padded input    
yzp= np.concatenate([y,np.zeros(Nfft-n_x)]) # zero-padded input    

X = np.fft.fft(xzp)   # input spectrum
Y = np.fft.fft(yzp)   # output spectrum
Sxx = X.conjugate() * X # PSD of X
Sxy = X.conjugate() * Y # CSD of XY
Rxx=np.fft.ifft((Sxx)) # Auto C of X
Ryx=np.fft.ifft((Sxy)) # Crocc C of XY
# Transfer function (double check!)
H_check=Sxy/Sxx
# Estimation of the impulse response
h_check=np.real(np.fft.ifft(H_check))
print('Estimated h in Freq Domain')
print(h_check)


#%% Measure cross coherence between probes 

plt.close('all')
# Plot The coherency
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Cxy = signal.coherence(P2_U, P3_U, Fs, nperseg=N_P)
plt.plot(f, 20*np.log10(Cxy),label='$C_{x,y}$')
plt.xlim([100,1500])

# Check the possible impulse response that could make the prediction 
# in the first

# Test: use the first 2000 points to get h.
# Then, try to predict the system

x_1=P2_U[0:2000]
y_1=P3_U[0:2000]

# Estimate h function: Very unstable!
def Estimate_h(x_1,y_1):
    # Get Spectra
    X = np.fft.fft(x_1)   # input spectrum
    Y = np.fft.fft(y_1)   # output spectrum
    # Compute PSDs
    Sxx = X.conjugate() * X # PSD of X
    Sxy = X.conjugate() * Y # CSD of XY
    # Transfer function (double check!)
    H_check=Sxy/Sxx
    h_estimate=np.real(np.fft.ifft(H_check))
    return h_estimate

h_estimate=Estimate_h(x_1,y_1);

P3_UP=signal.lfilter(h_estimate, 1, x_1)

plt.plot(P3_U)
plt.plot(P3_UP)

recovered, remainder = signal.deconvolve(x_1,y_1)

# # We now look for a way to compute the convolution using matrices:
# from scipy.linalg import convolution_matrix
# h=np.array([1, -1,-1, 0, 0,0])
# L=11
# A = convolution_matrix(h,L, mode='same')  
# print(A)  
# C= Circulant_CC(h,8)
# print(C) 


# x = np.array([1,200,0,4,1,4])
# C=Circulant_CC(h,len(x))
# print(C.dot(x))
# C=Circulant_CC(x,len(h))
# print(C.dot(h))
# print(np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(h,len(x)))))



# #%% Now we can compute the output with a circulant matrix
# # As manipulating system with take a moving average with
# h=np.ones(3)/3
# x=P2_U
# # We padd this to be able to work in the freq domain neatly:
# h=np.concatenate([h,np.zeros(len(x)-3)]) # zero-padded input   
# R_yx2=np.real(np.fft.ifft(np.fft.fft(Rxx)*np.fft.fft(h,len(x))))
# CC=Circulant_CC(Rxx,len(Rxx))
# R_yx3=CC.dot(h)

# plt.plot(R_yx2)
# plt.plot(R_yx3)

# y22=Circulant_CC(x,len(x)).dot(h)
# y23=Circulant_CC(h,len(h)).dot(x)


# def nextpow2(x):
#     """ Return N, the first power of 2 such that 2**N >= |x|"""
#     return int(np.ceil(np.log2(abs(x))))

# # We test first a crude approach in the freq domain
# n_x=len(x)
# #Nfft = 2 **nextpow2(n_x) # FFT wants power of 2
# Nfft=n_x



# Rxy2=np.fft.ifft(H*Sxx)
# plt.plot(Ryx)
# plt.plot(Ryx2)


# Hxy = Sxy / Sxx   # should be the freq. response
# hxy = np.fft.ifft(Hxy).real    # should be the imp. response
# print(hxy)

# plt.plot(f_fft,np.log(abs(Sxx)))
# plt.plot(f_fft,np.log(abs(Sxy)))
# plt.plot(f_fft,np.log(abs(Hxy)))



# # # Plot estimated versus true system
# # plt.plot(f_fft, 20 * np.log10(abs(Rxx)), 'r',label='|H(f)|')
# # plt.xlim([100,1500])

# # plt.plot(w/np.pi*Fs/2, 20 * np.log10(abs(H)), 'b',label='|H(f)|')






