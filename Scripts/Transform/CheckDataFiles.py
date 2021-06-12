"""
.. module:: CheckDataFiles

CoeckDataFiles
*************

:Description: CheckDataFiles

    

:Authors: bejar
    

:Version: 

:Created on: 15/01/2019 8:23 

"""
import glob
import numpy as np
import argparse
__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cols', help='number of expected columns', type=int, default=7)
    parser.add_argument('--sec', help='Pattern for the section of files', default='0')
    args = parser.parse_args()

    lfiles = glob.glob(f'{args.sec}-*.npy')

    for file in lfiles:
       data = np.load(file)
       if data.shape[1] != args.cols:
           print(file, data.shape)
