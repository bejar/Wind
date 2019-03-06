"""
.. module:: TestData

TestData
*************

:Description: TestData

    

:Authors: bejar
    

:Version: 

:Created on: 26/02/2019 6:37 

"""

import numpy as np


__author__ = 'bejar'


if __name__ == '__main__':
    a = np.load('/home/bejar/storage/Data/Wind/files/190/190-95125-12.npy')

    print(a.shape)

    print(np.mean(a, axis=0))

