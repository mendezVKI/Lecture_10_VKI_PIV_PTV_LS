# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:54:45 2021

@author: admin
"""


# Load the data from the probes

import numpy as np

import matplotlib.pyplot as plt

import os

import shutil

from Functions_Chapter_10 import Plot_Field_TEXT_Cylinder
FOLDER='DATA_CYLINDER'
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 

# Define the location of the probes in the grid
i1,j1=2,4; xp1=Xg[i1,j1]; yp1=Yg[i1,j1]
i2,j2=14,18; xp2=Xg[i2,j2]; yp2=Yg[i2,j2]
i3,j3=23,11; xp3=Xg[i3,j3]; yp3=Yg[i3,j3]
# Prepare the time axis (take only the first second!)
n_t=3000; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 
# Initialize Probes P1,P2,P3
P1=np.zeros(n_t);P2=np.zeros(n_t);P3=np.zeros(n_t);
# number of velocity points 
#(from Plot_Field_TEXT_Cylinder)
nxny=int(n_s/2) 
# Get number of columns and rows
n_x,n_y=np.shape(Xg)

# Initialize the probes
P1_U=np.zeros(n_t,dtype="float32") 
P1_V=np.zeros(n_t,dtype="float32") 
P2_U=np.zeros(n_t,dtype="float32") 
P2_V=np.zeros(n_t,dtype="float32") 
P3_U=np.zeros(n_t,dtype="float32") 
P3_V=np.zeros(n_t,dtype="float32") 

# Prepare the snapshot matrices U and V
D_U=np.zeros((n_s//2,n_t),dtype="float32") 
D_V=np.zeros((n_s//2,n_t),dtype="float32") 


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
  # Assign the vector like snapshots to snapshot matrices
  D_U[:,k]=V_X; D_V[:,k]=V_Y
  # Sample the U components
  P1_U[k]=V_X_g[i1,j1];P2_U[k]=V_X_g[i2,j2];P3_U[k]=V_X_g[i3,j3];
  # Sample the V Components
  P1_V[k]=V_Y_g[i1,j1];P2_V[k]=V_Y_g[i2,j2];P3_V[k]=V_Y_g[i3,j3];
  
  print('Loading Step '+str(k+1)+'/'+str(n_t)) 




#%% Save the data for the other exercises
# The exercise could start from here by loading the data without having
# to re-sample things.

np.savez('Sampled_PROBES',P1_U=P1_U,P2_U=P2_U,P3_U=P3_U,
                          P1_V=P1_V,P2_V=P2_V,P3_V=P3_V)


np.savez('Snapshot_Matrices',D_U=D_U,D_V=D_V,Xg=Xg,Yg=Yg)


Folder_OUT='C:\\Users\\mendez\\OneDrive - vki.ac.be\\Mendez_Contribution\\Chap10\\Figures\\'

# Question 1: Plot the mean field and the TI
# Having the snapshot matrices, this is a one line as in Ex1:
    
# U Component
mu_U=np.mean(D_U,axis=1) # Mean U
sigma_U=np.std(D_U,axis=1) # sigma_U
# V component
mu_V=np.mean(D_V,axis=1) # Mean V
sigma_V=np.std(D_V,axis=1) # sigma_U
    
# For plotting the mean field, we use the function in the provided module   
from  Functions_Chapter_10 import Plot_Field_Cylinder
from  Functions_Chapter_10 import Plot_Scalar_Field_Cylinder

Plot_Field_Cylinder(X_S,Y_S,mu_U,mu_V,True,2,0.8)
Name='Mean_Ex2.png'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory

# The mean velocity is 
U_Infty=np.max(np.sqrt(mu_U**2+mu_V**2));

Scalar=np.sqrt(sigma_U**2+sigma_V**2)/U_Infty
Plot_Scalar_Field_Cylinder(X_S,Y_S,mu_U,mu_V,Scalar,True,2,0.8)
Name='TI_Ex2.png'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



data = np.load('Sampled_PROBES.npz')
n_T=3000
P1_U=data['P1_U'][0:n_T]; 
P2_U=data['P2_U'][0:n_T];  
P3_U=data['P3_U'][0:n_T]; 

P1_V=data['P1_V'][0:n_T]; 
P2_V=data['P2_V'][0:n_T]; 
P3_V=data['P3_V'][0:n_T]; 

n_t=n_T; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

#%% Convergence of the statistics
n_T_v=np.round(np.logspace(1,3,num=41))
mu_1=np.zeros(len(n_T_v))
sigma_1=np.zeros(len(n_T_v))
mu_3=np.zeros(len(n_T_v))
sigma_3=np.zeros(len(n_T_v))

# Loop over all n_R's.
for n in range(len(n_T_v)):
    # show progress
    print('Computing n='+str(n)+' of '+str(len(n_T_v)))
    n_t=int(n_T_v[n]) # Define the number of ensambles
    # Compute the time averages of the Vel Magnitude
    mu_1[n]=np.mean(np.sqrt(P1_U[0:n_t]**2+P1_V[0:n_t]**2))
    mu_3[n]=np.mean(np.sqrt(P3_U[0:n_t]**2+P3_V[0:n_t]**2))
    # Compute the Std of the Vel Magnitude
    sigma_1[n]=np.std(np.sqrt(P1_U[0:n_t]**2+P1_V[0:n_t]**2))
    sigma_3[n]=np.std(np.sqrt(P3_U[0:n_t]**2+P3_V[0:n_t]**2))



fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
plt.plot(n_T_v,mu_1,'ko:',label='P1')
plt.plot(n_T_v,mu_3,'rs:',label='P3')
ax.set_xscale('log')
plt.xlabel('$n_t$',fontsize=18)
plt.ylabel('$\mu_U$',fontsize=18)
plt.legend()
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex2_Mean_Conv.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
plt.plot(n_T_v,sigma_1,'ko:', label='P1')
plt.plot(n_T_v,sigma_3,'rs:', label='P3')
ax.set_xscale('log')
plt.xlabel('$n_t$',fontsize=18)
plt.ylabel('$\sigma_U$',fontsize=18)
plt.legend()
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex2_Sigma_Conv.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory






#%% Correlation Analysis
def R_UW_C(u,v):
    # Perform the correlation in the Freq Domain
    RUU=np.fft.ifft(np.abs(np.fft.fft(u)*np.fft.fft(v))).real 
    # Normalize and remove the mean 
    c=(RUU/len(u)-(np.mean(u)*np.mean(v)))/(np.std(u)*np.std(v))
    return c[:len(u)//2]

from scipy import signal

def R_UW_L(u,v):
    # Call to the scipy function correlate:
    RUU = signal.correlate(u-np.mean(u), v-np.mean(v),\
                       mode='same',method='auto')/len(u)/(np.std(u)*np.std(v))
    return RUU[RUU.size//2:]

auto_Co_C=R_UW_C(P1_U, P1_U)
auto_Co_L=R_UW_L(P1_U, P1_U)


fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure

plt.plot(t[:len(t)//2],R_UW_C(P1_U, P1_U),'ko:',label='P1')

auto_Co_C=R_UW_C(P3_U, P3_U)
plt.plot(t[:len(t)//2],R_UW_C(P3_U, P3_U),'rs:',label='P3')

#plt.plot(t[:len(t)//2],auto_Co_L,'bs:')
plt.xlim([0,0.01])
plt.ylim([-0.75,1])
plt.xlabel('$\\tau [s]$',fontsize=18)
plt.ylabel('$\\tilde{R}_{UU}(\\tau)/\sigma^2_U$',fontsize=18)
plt.legend()
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex2_Auto_1_2.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



auto_Co_C=R_UW_C(P3_U, P3_U)
auto_Co_L=R_UW_L(P3_U, P3_U)


plt.figure(1)
plt.plot(t[:len(t)//2],auto_Co_C,'ko:')
plt.plot(t[:len(t)//2],auto_Co_L,'bs:')

plt.xlim([0,0.01])



# Compute the Reynolds stresses:
u1_f=P1_U-np.mean(P1_U);  v1_f=P1_V-np.mean(P1_V)
U_f_1=np.vstack((u1_f, v1_f)).T
Re_1=1/len(P1_U)*U_f_1.T.dot(U_f_1)    



 
fig, ax = plt.subplots(figsize=(4, 4)) # This creates the figure
plt.plot(u1_f,v1_f,'ko')
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
plt.xlabel('$u-\\langle u \\rangle$',fontsize=18)
plt.ylabel('$v-\\langle v \\rangle$',fontsize=18)
plt.title('P1',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex_2_UV_1.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



# Compute the Reynolds stresses:
u3_f=P3_U-np.mean(P3_U);  v3_f=P3_V-np.mean(P3_V)
U_f_3=np.vstack((u3_f, v3_f)).T
Re_3=1/len(P3_U)*U_f_3.T.dot(U_f_3)    


fig, ax = plt.subplots(figsize=(4, 4)) # This creates the figure
plt.plot(u3_f,v3_f,'ko')
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
plt.xlabel('$u-\\langle u \\rangle$',fontsize=18)
plt.ylabel('$v-\\langle v \\rangle$',fontsize=18)
plt.title('P3',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex_2_UV_3.pdf'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Folder_OUT+Name) #  copy file to latex directory



np.savez('u_3_Fluctuations',u3_f=u3_f,t=t)


