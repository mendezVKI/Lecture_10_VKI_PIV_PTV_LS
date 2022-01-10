# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 20:15:57 2021

@author: Admin
"""


# In this tutoril implements a Ornstein-Uhlenbeck Process
# and study its autocorrelation. In particular, compute the
# Integral Time Scale of the samples and the ensamble average

# %% Step 1: Create the OU_Process

import numpy as np
import matplotlib.pyplot as plt
import os

#os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-darwin'


# Initial and final time
t_0=0; t_end=10
# Number of Samples
n_t=1001
# Process Parameters
kappa=1.2; theta=3; sigma=0.5
# Create the time scale
t=np.linspace(t_0,t_end,n_t); dt=t[2]-t[1]



def U_O_Process(kappa,theta,sigma,t):
    n_T=len(t) 
    # Initialize the output
    y=np.zeros(n_T)
    # Define Drift and Diffusion functions in the process
    drift= lambda y,t: kappa*(theta-y)
    diff= lambda y,t: sigma
    noise=np.random.normal(loc=0,scale=1,size=n_T)*np.sqrt(dt)

    # Solve Stochastic Difference Equation
    for i in range(1,n_T):
        y[i]=y[i-1]+drift(y[i-1],i*dt)*dt+diff(y[i-1],i*dt)*noise[i]
    return y

#%% Step two: Produce the Ensamble of n_lambda=50 realizations
n_r=100
U_N=np.zeros((n_t,n_r))

for l in range(n_r):
    U_N[:,l]=U_O_Process(kappa,theta,sigma,t)

import shutil # To copy the figures in the Latex folder
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)


fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(t,U_N[:,1])
plt.plot(t,U_N[:,10])
plt.plot(t,U_N[:,22])
plt.plot(t,U_N[:,55])
plt.xlabel('$t[s]$',fontsize=18)
plt.ylabel('$u_{n}$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name='Ex1_samples.png'
plt.savefig(Name, dpi=200) 
shutil.copyfile(Name,Name) #  copy file to latex directory





# Compute the sample mean and plot it
y_Mean=np.mean(U_N,axis=1)
y_STD=np.std(U_N,axis=1)

plt.plot(t,y_Mean)
plt.plot(t,y_Mean+y_STD,'r--')
plt.plot(t,y_Mean-y_STD,'r--')

plt.show()

# Compute the autocorrelations

# Definition not normalized
def autocorr1(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    # Take only half
    return r2[:len(x)//2]

# Definition normalized: This is a autocovariance!
def autocorr2(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    # Take only half
    return c[:len(x)//2]

def autocorr3(x):
    result = np.correlate(x, x, mode='same')
    return result[result.size//2:]

def autocorr4(x):
    result = np.correlate(x, x, mode='same')/len(x)/np.var(x)
    return result[result.size//2:]



def CC_Norm_zeroPAD(x,y):
    result = np.correlate(x, y, mode='same')/len(x)/(np.std(x)*np.std(y))
    return result[result.size//2:]

def CC_Norm_fft(x,y):
    r2=np.fft.ifft(np.abs(np.fft.fft(x)*np.fft.fft(y))).real 
    c=(r2/len(x)-(np.mean(x)*np.mean(y)))/(np.std(x)*np.std(y))
    return c[:len(x)//2]

 


x=Y_Ensamble[:,2]

# Test the two implementations

plt.plot(autocorr2(x))
plt.plot(autocorr4(x-np.mean(x)))






# Autocorrelation of one signal
Auto1=autocorr2(Y_Ensamble[:,2])
Auto2=autocorr2(Y_Ensamble[:,10])
Auto3=autocorr2(Y_Ensamble[:,50])

Autocorrelation_Ensamble=np.zeros((n_T/2,n_lambda))
# We can now take the ensamble autocorrelation

for l in range(n_lambda):
    Autocorrelation_Ensamble[:,l]=autocorr2(Y_Ensamble[:,l])
    

Autocorrelation_MEAN=np.mean(Autocorrelation_Ensamble,axis=1) 
Autocorrelation_std=np.std(Autocorrelation_Ensamble,axis=1) 


plt.figure(2)
plt.plot(t[:len(t)//2],Auto1)
plt.plot(t[:len(t)//2],Auto2)
plt.plot(t[:len(t)//2],Auto3)

plt.plot(t[:len(t)//2],Autocorrelation_MEAN)
plt.plot(t[:len(t)//2],Autocorrelation_MEAN+Autocorrelation_std,'r--')
plt.plot(t[:len(t)//2],Autocorrelation_MEAN-Autocorrelation_std,'r--')
































