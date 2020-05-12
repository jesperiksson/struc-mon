# -*- spektra -*-
import scipy.io as sio
from scipy import fftpack
import numpy as np
import array as ar
import h5py
import matplotlib.pyplot as plt

mat = h5py.File('our_measurements/healthy/100%/s90/75643.mat','r')

acc= mat['acc']
acc=acc[1,:]


Fs = 1000.0;                        # Sampling frequencies
T = 60.0                            # Total time
t_1= 42.0*3.6/100;                  # time to pass the bridge
t_limit=t_1/(1/Fs)+1
t_limit=int(t_limit)

# klipp acc1
acc1=acc[t_limit:]



# tidsvektor
t_steg=1/Fs                         # steglängd för tidsvektor 
t=ar.array('f',range(0, int(T/t_steg)+1))
for i in range(0, int(T/t_steg)+1):
    t[i]=i*t_steg
t=t[t_limit:]

# frekvensvektor
f_steg=1/(T-t_1)                    # steg mellan frekvenser i plot    

f=ar.array('f',range(0, int(Fs/f_steg)))
for i in range(0, int(Fs/f_steg)):
    f[i]=i*f_steg



# Matlab fourier copy
N=len(t)

X=fftpack.fft(acc1)
S = 2.0/N
L = S*abs(X)
plt.semilogy(f,L) #test plot
plt.show()

from scipy import find_peaks
peaks= sio.find_peaks(L)
plt.plot(f,L)
plt.plot(peaks, f[peaks], "x")
