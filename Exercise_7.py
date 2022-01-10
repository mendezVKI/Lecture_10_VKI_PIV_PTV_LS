# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:17:50 2021

@author: mendez
"""

from modulo.modulo import MODULO


import numpy as np
import matplotlib.pyplot as plt

import shutil


Folder_OUT='C:\\Users\\mendez\\OneDrive - vki.ac.be\\Mendez_Contribution\\Chap10\\Figures\\'

# Setting for the plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)

# Prepare the required data
n_t=13200; Fs=3000; dt=1/Fs
# Prepare the time axis
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# As input signal we take P2.
data = np.load('Snapshot_Matrices.npz')
Xg=data['Xg']; Yg=data['Yg']
D_U=data['D_U']; D_V=data['D_V']

# Assembly both components in one single matrix
D=np.concatenate([D_U,D_V],axis=0) # Reshape and assign
# # We remove the time average (for plotting purposes)
# D_MEAN=np.mean(D_M,1) # Temporal average (along the columns)
# D=D_M-np.array([D_MEAN,]*n_t).transpose() # Mean Removed
# del D_M
# del D_U; del D_V

# Shape of the grid
nxny=D.shape[0]//2; 
# Shape of the grid
n_y,n_x=np.shape(Xg)
# Debugging
# D=D[:,0:2000]
# Number of Modes searched for:
n_r=500

#%% Perform the mPOD with Modulo
# Initialize a 'MODULO Object'
m = MODULO(data=D,
           MEMORY_SAVING=True,
           n_Modes = n_r,
           N_PARTITIONS=11)
# Prepare (partition) the dataset
D = m._data_processing() # Note that this will cancel D!
# mPOD Settings
# Frequency splitting Vector
F_V=np.array([10,290,320,430,470])
# This vector collects the length of the filter kernels.
Nf=np.array([1201,801,801,801,801]); 
Keep=np.array([0,1,0,1,0])
Ex=1203; boundaries = 'nearest'; MODE = 'reduced'
# Compute the mPOD 
Phi_M, Psi_M, Sigmas_M = m.compute_mPOD(Nf, Ex, F_V,
                                        Keep, boundaries, 
                                        MODE, 1/Fs)


from Functions_Chapter_10 import plot_grid_cylinder_flow

# Prepare the frequencies 
Freqs = np.fft.fftfreq(n_t) * Fs  # Compute the frequency bins



# Plot the spatial structures and the spectra of the first 3 modes
for r in range(6):
 #%% Plot the spatial structure
 Phi_M_Mode_U=np.real(Phi_M[0:nxny,r]).reshape((n_x,n_y))
 Phi_M_Mode_V=np.real(Phi_M[nxny::,r]).reshape((n_x,n_y))
 # Assign to Vxg, Vyg which will be plot
 Vxg=Phi_M_Mode_U; Vyg=Phi_M_Mode_V
 plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
 Tit_string='$\phi_{\mathcal{M}'+str(r)+'}(\mathbf{x}_i)\}$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name='mPOD_Phi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory
 #%% Plot the spectral content of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 psi_hat=np.fft.fft(Psi_M[:,r]*Sigmas_M[r]/n_t)
 plt.plot(Freqs,20*np.log10(psi_hat),'ko')
 plt.xlim([0,1500])
 plt.ylim([-125,50])
 #ax.set_xscale('log')
 plt.xlabel('$ f_n [Hz]$',fontsize=18)
 plt.ylabel('$\mathcal{F}\{\psi_{\mathcal{M}r}\} [db]$',fontsize=18)
 Tit_string='$\mathcal{F}\{\psi\}_{\mathcal{M}'+str(r)+'}(f_n)\}$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name='mPOD_Psi_hat_'+str(r)+'R.pdf'
 plt.savefig(Name, dpi=200) 
 shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory
 #%% Plot the spectral content of the structure
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 plt.plot(t,Psi_M[:,r],'-')
 plt.xlabel('$ t_k [s]$',fontsize=18)
 plt.ylabel('$\psi_{\mathcal{M}r}$',fontsize=18)
 Tit_string='$\psi_{\mathcal{M}'+str(r)+'}(t_k)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name='mPOD_Psi_'+str(r)+'R.pdf'
 plt.savefig(Name, dpi=200) 
 shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


plt.close('all')




