"""
.. module:: Corr_test

Corr_test
*************

:Description: Corr_test

    Test of correlation values

:Authors: bejar
    

:Version: 

:Created on: 08/04/2019 9:49 

"""

import numpy as np
from Wind.Config.Paths import wind_data_path, wind_path, wind_NREL_data_path
__author__ = 'bejar'


if __name__ == '__main__':
    d = "11-5794-12"
    wind = np.load(wind_data_path + f"/{d}.npy")
    print(wind.shape)

    print(np.corrcoef(wind,rowvar=False))
    lag = 12
    cmat = np.zeros((2,2))
    for i in range(wind.shape[0]-lag):
        cmat +=np.corrcoef(wind[i:i+lag,0], wind[i:i+lag,5],rowvar=False)

    cmat/=cmat[0,0]
    print(cmat)

