# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:36:21 2021

@author: mendez
"""

# Functions for the Chapter 10

import numpy as np
from numpy import linalg as LA
from scipy.signal import firwin 
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

 
def Plot_Field_TEXT_Cylinder(File,Name_Mesh):  
   """
   This function plots the vector field from the TR-PIV in Exercise 4.
      
    :param File: Name of the file to load          
   
   """
   # We first perform a zero padding
   Name=File
   # Read data from a file
   DATA = np.genfromtxt(Name) # Here we have the four colums
   Dat=DATA[1:,:] # Here we remove the first raw, containing the header
   nxny=Dat.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
   n_s=2*nxny
   ## 1. Reconstruct Mesh from file Name_Mesh
   DATA_mesh=np.genfromtxt(Name_Mesh);
   Dat_mesh=DATA_mesh[1:,:]
   X_S=Dat_mesh[:,0];
   Y_S=Dat_mesh[:,1];
   # Reshape also the velocity components
   V_X=Dat[:,0] # U component
   V_Y=Dat[:,1] # V component
   # Put both components as fields in the grid
   Xg,Yg,Vxg,Vyg,Magn=Plot_Field_Cylinder(X_S,Y_S,V_X,V_Y,False,2,0.6)
   # Show this particular step
   fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
   # Or you can plot it as streamlines
   CL=plt.contourf(Xg,Yg,Magn,levels=np.arange(0,18,2))
   # One possibility is to use quiver
   STEPx=1;  STEPy=1
   plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
   plt.rc('text', usetex=True)      
   plt.rc('font', family='serif')
   plt.rc('xtick',labelsize=12)
   plt.rc('ytick',labelsize=12)
   fig.colorbar(CL,pad=0.05,fraction=0.025)
   ax.set_aspect('equal') # Set equal aspect ratio
   ax.set_xlabel('$x[mm]$',fontsize=13)
   ax.set_ylabel('$y[mm]$',fontsize=13)
   #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
   ax.set_xticks(np.arange(0,70,10))
   ax.set_yticks(np.arange(-10,11,10))
   ax.set_xlim([0,50])
   ax.set_ylim(-10,10)
   circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
   plt.gcf().gca().add_artist(circle)
   plt.tight_layout()
   plt.show()
   Name='Snapshot_Cylinder.png'
   plt.savefig(Name, dpi=200) 
   Name[len(Name)-12:len(Name)]+' Plotted'
   return n_s, Xg, Yg, Vxg, Vyg, X_S, Y_S


def Plot_Field_Cylinder(X_S,Y_S,V_X,V_Y,PLOT,Step,Scale):
  # Number of n_X/n_Y from forward differences
   GRAD_X=np.diff(Y_S) 
   #GRAD_Y=np.diff(Y_S);
   # Depending on the reshaping performed, one of the two will start with
   # non-zero gradient. The other will have zero gradient only on the change.
   IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
   nxny=X_S.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
   #n_s=2*nxny
   # Reshaping the grid from the data
   n_x=(nxny//(n_y)) # Carefull with integer and float!
   Xg=np.transpose(X_S.reshape((n_x,n_y)))
   Yg=np.transpose(Y_S.reshape((n_x,n_y))) # This is now the mesh! 60x114.
   Mod=np.sqrt(V_X**2+V_Y**2)
   Vxg=np.transpose(V_X.reshape((n_x,n_y)))
   Vyg=np.transpose(V_Y.reshape((n_x,n_y)))
   Magn=np.transpose(Mod.reshape((n_x,n_y)))
  
   if PLOT:
    # One possibility is to use quiver
    STEPx=Step;  STEPy=Step
    fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # Or you can plot it as streamlines
    CL=plt.contourf(Xg,Yg,Magn,levels=np.arange(0,18,2))
    # One possibility is to use quiver
    STEPx=1;  STEPy=1
    plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
    plt.rc('text', usetex=True)      
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig.colorbar(CL,pad=0.05,fraction=0.025)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(-10,11,10))
    ax.set_xlim([0,50])
    ax.set_ylim(-10,10)
    circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    plt.show()
       
   return Xg,Yg,Vxg,Vyg,Magn  
   
def Plot_Scalar_Field_Cylinder(X_S,Y_S,V_X,V_Y,Scalar,PLOT,Step,Scale):
  # Number of n_X/n_Y from forward differences
   GRAD_X=np.diff(Y_S) 
   #GRAD_Y=np.diff(Y_S);
   # Depending on the reshaping performed, one of the two will start with
   # non-zero gradient. The other will have zero gradient only on the change.
   IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
   nxny=X_S.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
   #n_s=2*nxny
   # Reshaping the grid from the data
   n_x=(nxny//(n_y)) # Carefull with integer and float!
   Xg=np.transpose(X_S.reshape((n_x,n_y)))
   Yg=np.transpose(Y_S.reshape((n_x,n_y))) # This is now the mesh! 60x114.
   Vxg=np.transpose(V_X.reshape((n_x,n_y)))
   Vyg=np.transpose(V_Y.reshape((n_x,n_y)))
   Magn=np.transpose(Scalar.reshape((n_x,n_y)))
  
   if PLOT:
    # One possibility is to use quiver
    STEPx=Step;  STEPy=Step
    fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # Or you can plot it as streamlines
    CL=plt.contourf(Xg,Yg,Magn*100,levels=np.arange(0,70,10))
    # One possibility is to use quiver
    STEPx=1;  STEPy=1
    plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
    plt.rc('text', usetex=True)      
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig.colorbar(CL,pad=0.05,fraction=0.025)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(-10,11,10))
    ax.set_xlim([0,50])
    ax.set_ylim(-10,10)
    circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    plt.show()
       
   return Xg,Yg,Vxg,Vyg,Magn  



def plot_grid_cylinder_flow(Xg,Yg,Vxg,Vyg):
 STEPx=1;  STEPy=1
 # This creates the figure
 fig, ax = plt.subplots(figsize=(6, 3)) 
 Magn=np.sqrt(Vxg**2+Vyg**2)
 # Plot Contour
 #CL=plt.contourf(Xg,Yg,Magn,levels=np.linspace(0,np.max(Magn),5))
 CL=plt.contourf(Xg,Yg,Magn,20,cmap='viridis',alpha=0.95)
 # One possibility is to use quiver
 STEPx=1;  STEPy=1
 plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
            Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
 plt.rc('text', usetex=True)      
 plt.rc('font', family='serif')
 plt.rc('xtick',labelsize=12)
 plt.rc('ytick',labelsize=12)
 #fig.colorbar(CL,pad=0.05,fraction=0.025)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xlabel('$x[mm]$',fontsize=13)
 ax.set_ylabel('$y[mm]$',fontsize=13)
 #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
 ax.set_xticks(np.arange(0,70,10))
 ax.set_yticks(np.arange(-10,11,10))
 ax.set_xlim([0,50])
 ax.set_ylim(-10,10)
 circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
 plt.gcf().gca().add_artist(circle)
 plt.tight_layout()
 plt.show()


# Taken from https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

def detrend(h):
    # This is a linear detrend operation
    n = len(h)
    t = np.arange(n)
    p = np.polyfit(t, h, 1)
    h_detrended = h - np.polyval(p, t)
    return h_detrended

def quadwin(n):
    """
    Quadratic (or "Welch") window
    """
    t = np.arange(n)
    win = 1 - ((t - 0.5 * n) / (0.5 * n)) ** 2
    return win

def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw

def spectrum2(h, dt=1, nsmooth=5):
    """
    Add simple boxcar smoothing to the raw periodogram.
    
    Chop off the ends to avoid end effects.
    """
    freqs, ps, psd = spectrum1(h, dt=dt)
    weights = np.ones(nsmooth, dtype=float) / nsmooth
    ps_s = np.convolve(ps, weights, mode='valid')
    psd_s = np.convolve(psd, weights, mode='valid')
    freqs_s = np.convolve(freqs, weights, mode='valid')
    return freqs_s, ps_s, psd_s


def PSD_Freq(h, dt=1, nsmooth=5):
    """
    Detrend and apply a quadratic window to the PSD
    # Returns both the Power spectral (ps) and the
    power spectral density
    """
    # Get length of the signal
    n = len(h)
    # Detrend the signal (h)
    h_detrended = detrend(h)
    # create the window
    winweights = quadwin(n)
    # tape on the sides
    h_win = h_detrended * winweights
    # Compute the the spectrum from fft
    freqs, ps, psd = spectrum2(h_win, dt=dt, nsmooth=nsmooth)
    
    # Compensate for the energy suppressed by the window.
    psd *= n / (winweights**2).sum()
    ps *= n**2 / winweights.sum()**2
    
    return freqs, ps, psd

