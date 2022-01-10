# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:35:36 2021

@author: mendez
"""

import shutil # To copy the figures in the Latex folder

Folder_OUT='C:\\Users\\mendez\\OneDrive - vki.ac.be\\Mendez_Contribution\\Chap10\\Figures\\'

# Script 1: Get the Data
import urllib.request
print('Downloading Data PIV for Chapter 10...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r'); 
zf.extractall('./DATA_CYLINDER'); zf.close()


# Script 2: Print a Field
import os
import numpy as np

import matplotlib.pyplot as plt
from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder
FOLDER='DATA_CYLINDER'
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'

n_x,n_y=np.shape(Xg)

# Select the positions for the probes:
i1,j1=2,4; xp1=Xg[i1,j1]; yp1=Yg[i1,j1]
i2,j2=14,18; xp2=Xg[i2,j2]; yp2=Yg[i2,j2]
i3,j3=23,11; xp3=Xg[i3,j3]; yp3=Yg[i3,j3]

# Plot the location of the probes
plt.plot(xp1,yp1,'ro',markersize=10)
plt.annotate(r'P1', xy=(9, -9),color='black',\
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
plt.plot(xp2,yp2,'ro',markersize=10)
plt.annotate(r'P2', xy=(19, 4),color='black',\
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))
plt.plot(xp3,yp3,'ro',markersize=10)
plt.annotate(r'P3', xy=(27, -3),color='black',\
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9))

plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


#%% Get Time Series
# Import numpy
import numpy as np
# Define the location of the probes in the grid
i1,j1=2,4; xp1=Xg[i1,j1]; yp1=Yg[i1,j1]
i2,j2=14,18; xp2=Xg[i2,j2]; yp2=Yg[i2,j2]
i3,j3=23,11; xp3=Xg[i3,j3]; yp3=Yg[i3,j3]
# Prepare the time axis
n_t=13200; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 
# Initialize Probes P1,P2,P3
P1=np.zeros(n_t);P2=np.zeros(n_t);P3=np.zeros(n_t);
# number of velocity points 
#(from Plot_Field_TEXT_Cylinder)
nxny=int(n_s/2) 

# We extract the components separately also
P1_U=np.zeros(n_t)
P1_V=np.zeros(n_t)
P2_U=np.zeros(n_t)
P2_V=np.zeros(n_t)
P3_U=np.zeros(n_t)
P3_V=np.zeros(n_t)


for k in range(0,n_t):
  # Name of the file to read
  Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat' 
  # Read data from a file
  # Here we have the two colums
  DATA = np.genfromtxt(Name,usecols=np.arange(0,2),max_rows=nxny+1) 
  Dat=DATA[1:,:] # Remove the first raw with the header
  # Get U component and reshape as grid
  V_X=Dat[:,0]; V_X_g=np.reshape(V_X,(n_y,n_x))
  # Get V component and reshape as grid
  V_Y=Dat[:,1]; V_Y_g=np.reshape(V_Y,(n_y,n_x))
  # Get velocity magnitude
  V_MAGN=np.sqrt(V_X_g**2+V_Y_g**2)
  # Sample the three probes
  P1[k]=V_MAGN[i1,j1];P2[k]=V_MAGN[i2,j2];P3[k]=V_MAGN[i3,j3];

  print('Loading Step '+str(k+1)+'/'+str(n_t)) 





#%% Plots from Section 1
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(t,P1)
plt.xlabel('$t[s]$',fontsize=18)
plt.ylabel('$u_{1}[m/s]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='U_1.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(t,P2)
plt.xlabel('$t[s]$',fontsize=18)
plt.ylabel('$u_{2}[m/s]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='U_2.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(t,P3)
plt.xlabel('$t[s]$',fontsize=18)
plt.ylabel('$u_{3}[m/s]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='U_3.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



# Histograms
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.hist(P1,bins=1000,density=True)
plt.xlabel('$u_{1}[m/s]$',fontsize=18)
plt.ylabel('$f_{u1}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='hU_1.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.hist(P2,bins=1000,density=True)
plt.xlabel('$u_{2}[m/s]$',fontsize=18)
plt.ylabel('$f_{u2}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='hU_2.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.hist(P3,bins=1000,density=True)
plt.xlabel('$u_{3}[m/s]$',fontsize=18)
plt.ylabel('$f_{u3}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='hU_3.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory
plt.close('all')


#%% Plot the spectrograms
from scipy import signal
N_P=2048*2
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Pxx_den = signal.welch(P1-np.mean(P1), Fs, nperseg=N_P,scaling='density')
plt.plot(f, 20*np.log10(Pxx_den))
plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')
# # xticks=np.array([100, 200, 300, 1000,1500])
# # ax.xaxis.set_ticklabels( xticks )
# ax.set_xticks([100,200, 300, 500, 1000])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$PSD_{U1} [dB]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='PSD_U_1.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

plt.close('all')

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Pxx_den = signal.welch(P2-np.mean(P2), Fs, nperseg=N_P)
plt.plot(f, 20*np.log10(Pxx_den))
plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')
# # xticks=np.array([100, 200, 300, 1000,1500])
# # ax.xaxis.set_ticklabels( xticks )
# ax.set_xticks([100,200, 300, 500, 1000])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$PSD_{U2} [dB]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='PSD_U_2.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

plt.close('all')


fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
f, Pxx_den = signal.welch(P3-np.mean(P3), Fs, nperseg=N_P)
plt.plot(f, 20*np.log10(Pxx_den))
plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')
# xticks=np.array([100, 200, 300, 1000,1500])
# ax.xaxis.set_ticklabels( xticks )
# ax.set_xticks([100,200, 300, 500, 1000])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('$f [Hz]$',fontsize=18)
plt.ylabel('$PSD_{U3} [dB]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='PSD_U_3.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

plt.close('all')

## Same but FFT based
# n=len(P3)
# Freqs=np.fft.fftshift(np.fft.fftfreq(n))*Fs # This is np tool for freq axis
# PSD=np.abs(np.fft.fft(P3-np.mean(P3)))**2/(n**2)
# fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
# plt.plot(Freqs,np.fft.fftshift(PSD))
# plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')
# # xticks=np.array([100, 200, 300, 1000,1500])
# # ax.xaxis.set_ticklabels( xticks )
# ax.set_xticks([100,200, 300, 500, 1000])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# Name='PSD_U_3.pdf'
# plt.savefig(Name, dpi=200) 
# plt.close('all')



# from Functions_Chapter_10 import PSD_Freq
# # Version of the welch spectra in the frequency domain
# freqs1, ps1, psd1 = PSD_Freq(P3-np.mean(P3), \
#                              dt=1/Fs,nsmooth=10)
# df=1/(n/Fs)

# fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
# plt.plot(freqs1,20*np.log10(psd1))
# plt.xlim([100,1500])
# ax.set_yscale('log')
# ax.set_xscale('log')




# get the spectra from FFT


# fft_abs=1/n*np.fft.fftshift(np.abs(np.fft.fft(x-np.mean(x))))
# plt.plot(Freqs,fft_abs**2)

# plt.plot(Freqs,np.fft.fftshift(PSD))





# # Check the Pwelch spectrogram
# x=2*np.sin(2*np.pi*50*t)
# # plt.plot(t,x,'ko:')

# N_P=1024
# f, Pxx_den = signal.welch(x-np.mean(x), Fs,\
#                           nperseg=N_P,scaling='density')
# W=np.hamming(N_P)    
# plt.plot(f, Pxx_den)
# print(np.max(Pxx_den))




