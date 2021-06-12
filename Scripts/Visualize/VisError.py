"""
.. module:: VisError

VisError
*************

:Description: VisError

    

:Authors: bejar
    

:Version: 

:Created on: 11/02/2020 12:27 

"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
__author__ = 'bejar'


f = h5py.File('/home/bejar/PycharmProjects/Wind/errorsMLPS2S-S11-5793-12-1-12-R0.hdf5', 'r+')

val_y = f['/errors/val_yu']
val_yp = f['/errors/val_ypu']

print(val_y.shape, val_yp.shape)

err = val_y[()] - val_yp[()]

#err *= err
#err = np.abs(err)

nvals = 8000
prop = 100
pixels = np.zeros((12*prop, nvals))



for i in range(12):
    for j in range(prop):
        pixels[(i*prop) + j,:] = err[:nvals,i]

pixels[0,0] = -np.max(pixels)

fig = plt.figure(figsize=(10,5))
ax = plt.axes()
im = ax.imshow(pixels,cmap="seismic")
plt.colorbar(im, fraction=0.006, pad=0.04)
plt.show()
