# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:25:57 2021

@author: mendez
"""

from modulo.modulo import MODULO


import numpy as np
import matplotlib.pyplot as plt


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


# Shape of the grid
nxny=D.shape[0]//2; 
# Shape of the grid
n_y,n_x=np.shape(Xg)


# Debugging
# D=D[:,0:2000]

# Number of Modes searched for:
n_r=500

#%% Perform the POD with Modulo
# Initialize a 'MODULO Object'
m = MODULO(data=D,
           MEMORY_SAVING=True,
           n_Modes = n_r,
           N_PARTITIONS=11)
# Prepare (partition) the dataset
D = m._data_processing() # Note that this will cancel D!
# Compute the POD
Phi_P, Psi_P, Sigma_P = m.compute_POD()

import shutil

# Amplitudes of the POD modes
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(np.linspace(1,n_r,n_r),Sigma_P/Sigma_P[0],'ko:')
ax.set_xscale('log')
plt.xlim([0,n_r])
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{P}r}/(\sigma_{\mathcal{P}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='POD_R.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


from Functions_Chapter_10 import plot_grid_cylinder_flow

# Prepare the frequencies 
Freqs = np.fft.fftfreq(n_t) * Fs  # Compute the frequency bins



# Plot the spatial structures and the spectra of the first 3 modes
for r in range(3):
 #%% Take the mode's structure
 Phi_P_Mode_U=np.real(Phi_P[0:nxny,r]).reshape((n_x,n_y))
 Phi_P_Mode_V=np.real(Phi_P[nxny::,r]).reshape((n_x,n_y))
 # Assign to Vxg, Vyg which will be plot
 Vxg=Phi_P_Mode_U; Vyg=Phi_P_Mode_V
 plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
 Tit_string='$\phi_{\mathcal{P}'+str(r)+'}(\mathbf{x}_i)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name='POD_Phi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory
 #%% Plot the spectral content of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 psi_hat=np.fft.fft(Psi_P[:,r]*Sigma_P[r]/n_t)
 plt.plot(Freqs,20*np.log10(psi_hat),'ko')
 plt.xlim([0,1500])
 plt.ylim([-125,50])
 #ax.set_xscale('log')
 plt.xlabel('$ f_n [Hz]$',fontsize=18)
 plt.ylabel('$\mathcal{F}\{\psi_{\mathcal{P}r}\} [db]$',fontsize=18)
 Tit_string='$\mathcal{F}\{\psi\}_{\mathcal{P}'+str(r)+'}(f_n)\}$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name='POD_Psi_hat_'+str(r)+'R.pdf'
 plt.savefig(Name, dpi=200) 
 shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


plt.close('all')


#%% Plot the trajectory

from mpl_toolkits.mplot3d import Axes3D

# look at the orbit we saw before:
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig, elev=-150, azim=48)
ax.dist = 12
ax.scatter(Psi_P[1:13000:5, 0],
           Psi_P[1:13000:5, 1],
           Psi_P[1:13000:5, 2], 'ko')
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


ax.set_xlabel('$\psi_{\mathcal{P}1}$',fontsize=16)
ax.set_ylabel('$\psi_{\mathcal{P}2}$',fontsize=16)
ax.set_zlabel('$\psi_{\mathcal{P}3}$',fontsize=16)
#plt.tight_layout()
Name='3D_POD_Cylinder_Manifold.pdf'
plt.savefig(Name, dpi=300) 
#plt.show()
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory





