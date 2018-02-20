"""
.. module:: Winpred

Winpred
*************

:Description: Winpred

    

:Authors: bejar
    

:Version: 

:Created on: 28/06/2017 10:37 

"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from Wind.Config import wind_data_path, wind_data_ext, wind_path
import statsmodels.api as sm
from __future__ import division

__author__ = 'bejar'


nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/0/0.nc", 'r')

tseries = nc_fid.variables['wind_speed'][0:1000]

