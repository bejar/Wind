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
    # print(nf, lt, ln)
    lcoords.append([float(lt), float(ln)])

vcoord = np.array(lcoords)
print vcoord.shape

np.save('../Data/coords.npy', vcoord)