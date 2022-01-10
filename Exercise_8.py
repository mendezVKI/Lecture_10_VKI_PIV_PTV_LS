# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:37:34 2021

@author: Miguel Mendez
"""


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

# Shape of the grid
nxny=D.shape[0]//2; 
# Shape of the grid
n_y,n_x=np.shape(Xg)
# Debugging
# D=D[:,0:2000]

D_s=D.T
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel="rbf", n_components=3,fit_inverse_transform=True, 
                 alpha=1,gamma=None)
Psi_K = kpca.fit_transform(D_s)


from mpl_toolkits.mplot3d import Axes3D

# look at the orbit we saw before:
fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig, elev=-150, azim=48)
ax.dist = 13
ax.scatter(Psi_K[0:n_t:2,1],
           -Psi_K[0:n_t:2,2],
           Psi_K[0:n_t:2,0], 'ko')
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)


ax.set_zlabel('$\psi_{\mathcal{K}1}$',fontsize=16)
ax.set_xlabel('$\psi_{\mathcal{K}2}$',fontsize=16)
ax.set_ylabel('$\psi_{\mathcal{K}3}$',fontsize=16)
#plt.tight_layout()
Name='3D_kPOD_Cylinder_Manifold.pdf'
plt.savefig(Name, dpi=300) 
#plt.show()
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory


X_rec_kpca = kpca.inverse_transform(Psi_K )
Error_kpca=np.linalg.norm(D_s-X_rec_kpca)/np.linalg.norm(D_s)
print(Error_kpca)









