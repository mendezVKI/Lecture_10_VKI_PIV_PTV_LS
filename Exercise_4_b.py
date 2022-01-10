# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:39:00 2021

@author: admin
"""

# We compute the DFT modes for the cylinder test case

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
n_t=3000; Fs=3000; dt=1/Fs
# Prepare the time axis
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# As input signal we take P2.
data = np.load('Snapshot_Matrices.npz')
Xg=data['Xg']; Yg=data['Yg']
D_U=data['D_U']; D_V=data['D_V']
# Assembly both components in one single matrix
D_M=np.concatenate([D_U,D_V],axis=0) # Reshape and assign
# We remove the time average (for plotting purposes)
D_MEAN=np.mean(D_M,1) # Temporal average (along the columns)
D=D_M-np.array([D_MEAN,]*n_t).transpose() # Mean Removed
del D_M



#%% DFT Using MODULO
from modulo.modulo import MODULO
# Prepare MODULO Object
m = MODULO(data=D)
# Compute the DFT
Sorted_Freqs, Phi_F, SIGMA_F = m.compute_DFT(Fs)

Sorted_Sigmas=np.diag(SIGMA_F)
# Check the spectra of the DFT
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sorted_Freqs,
         20*np.log10((Sorted_Sigmas/np.sqrt((D.shape[0]*n_t)))),'ko')
plt.xlim([0,1500])
plt.ylim([-50,1])
plt.xlabel('$ f [Hz]$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{F}r}/(n_s n_r)$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DFT_Spectra.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


# Check the spectra of the DFT
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sorted_Sigmas[0:n_t-1]/np.max(Sorted_Sigmas),'ko:')
ax.set_yscale('log')
plt.xlim([0,n_t-1])
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{F}r}/(\sigma_{\mathcal{F}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DFT_R.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

from Functions_Chapter_10 import plot_grid_cylinder_flow


# The modes come in pair. We plot the spatial structure of the second mode
nxny=D.shape[0]//2; 
# Shape of the grid
n_y,n_x=np.shape(Xg)
# Extrac the real part

Vxg=D_MEAN[0:nxny].reshape((n_x,n_y))
Vyg=D_MEAN[nxny::].reshape((n_x,n_y))
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)


r=0; 
# Take the real part of mode r
Phi_F_Mode_U_r=np.real(Phi_F[0:nxny,r]).reshape((n_x,n_y))
Phi_F_Mode_V_r=np.real(Phi_F[nxny::,r]).reshape((n_x,n_y))
# Take the imaginary part of mode r
Phi_F_Mode_U_i=np.imag(Phi_F[0:nxny,r]).reshape((n_x,n_y))
Phi_F_Mode_V_i=np.imag(Phi_F[nxny::,r]).reshape((n_x,n_y))

Vxg=Phi_F_Mode_U_r
Vyg=Phi_F_Mode_V_r
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
plt.title('$\mathcal{R}\{ \Phi_{\mathcal{F}1}(\mathbf{x}_i)\}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DFT_Mode_R.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



Vxg=Phi_F_Mode_U_i
Vyg=Phi_F_Mode_V_i
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
plt.title('$\mathcal{I}\{ \Phi_{\mathcal{F}1}(\mathbf{x}_i)\}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DFT_Mode_I.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory








