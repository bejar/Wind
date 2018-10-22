"""
.. module:: processcoords

processcoords
*************

:Description: processcoords

    

:Authors: bejar
    

:Version: 

:Created on: 05/06/2018 14:33 

"""

import numpy as np

__author__ = 'bejar'


cfile = '/home/bejar/storage/Data/Wind/Results/Coords.csv'

lcoords = []

file = open(cfile,'r')

for l in file:
    lt, ln, nf = l.split(',')
    nf = nf.split('/')[-1].split('.')[0]
    lcoords.append([int(nf), float(lt), float(ln)])

lcoords = sorted(lcoords, key=lambda x : x[0])

for i in range(1000):
    print(lcoords[i])

vcoord = np.array(lcoords)


print(vcoord[:,1:].shape)

np.save('../Data/coords.npy', vcoord[:,1:])