# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:17:50 2021

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
n_t=3000; Fs=3000; dt=1/Fs
# Prepare the time axis
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# As input signal we take P2.
data = np.load('Snapshot_Matrices.npz')
Xg=data['Xg']; Yg=data['Yg']
D_U=data['D_U'][:,0:n_t]; D_V=data['D_V'][:,0:n_t]
# Assembly both components in one single matrix
D_M=np.concatenate([D_U,D_V],axis=0) # Reshape and assign
# We remove the time average (for plotting purposes)
D_MEAN=np.mean(D_M,1) # Temporal average (along the columns)
D=D_M-np.array([D_MEAN,]*n_t).transpose() # Mean Removed
del D_M


# Shape of the grid
nxny=D.shape[0]//2; 
# Shape of the grid
n_y,n_x=np.shape(Xg)
# Debugging
# D=D[:,0:2000]
# Number of Modes searched for:
n_r=500

#%% Perform the DMD with Modulo
# Initialize a 'MODULO Object'
m = MODULO(data=D,
           MEMORY_SAVING=True,
           n_Modes = n_r,
           N_PARTITIONS=11)
# Prepare (partition) the dataset
D = m._data_processing() # Note that this will cancel D!
# Compute the POD
Phi_D, Lambda, freqs, a0s = m.compute_DMD(F_S=3000)

# This creates the figure
fig, ax = plt.subplots(figsize=(4, 4)) 
plt.plot(np.imag(Lambda),np.real(Lambda),'ko')
circle=plt.Circle((0,0),1,color='b', fill=False)
ax.add_patch(circle)
ax.set_aspect('equal')
ax.set_xlabel('$\mathcal{R}\{\lambda_{\mathcal{D}r}\}$',fontsize=16)
ax.set_ylabel('$\mathcal{R}\{\lambda_{\mathcal{D}r}\}$',fontsize=16)
plt.tight_layout()
Name='lambdas_D.pdf'
plt.savefig(Name, dpi=300) 
#plt.show()
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


fig, ax = plt.subplots(figsize=(6, 4)) # This creates the figure
plt.plot(freqs,np.abs(Lambda),'ko') 
ax.set_xlabel('$f_n [Hz]$',fontsize=16)
ax.set_ylabel('$||\lambda_{\mathcal{D}r}||$',fontsize=16)
plt.tight_layout()
Name='lambdas_D_Freqs.pdf'
plt.savefig(Name, dpi=300) 
#plt.show()
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


# Look for the mode with abs(lambda) approx 1: these are the first two
from Functions_Chapter_10 import plot_grid_cylinder_flow

# The modes come in pair. We plot the spatial structure of the firt modes

r=0; 
# Take the real part of mode r
Phi_D_Mode_U_r=np.real(Phi_D[0:nxny,r]).reshape((n_x,n_y))
Phi_D_Mode_V_r=np.real(Phi_D[nxny::,r]).reshape((n_x,n_y))
# Take the imaginary part of mode r
Phi_D_Mode_U_i=np.imag(Phi_D[0:nxny,r]).reshape((n_x,n_y))
Phi_D_Mode_V_i=np.imag(Phi_D[nxny::,r]).reshape((n_x,n_y))

Vxg=Phi_D_Mode_U_r
Vyg=Phi_D_Mode_V_r
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
plt.title('$\mathcal{R}\{ \Phi_{\mathcal{D}1}(\mathbf{x}_i)\}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DMD_Mode_R.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



Vxg=Phi_D_Mode_U_i
Vyg=Phi_D_Mode_V_i
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
plt.title('$\mathcal{I}\{ \Phi_{\mathcal{D}1}(\mathbf{x}_i)\}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='DMD_Mode_I.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


