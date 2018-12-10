"""
.. module:: GenerateCoordinatesFile

GenerateCoordinatesFile
*************

:Description: GenerateCoordinatesFile

    Extracts the longitude and latitude from the files and saves in the file coords.npy

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 10:38 

"""

__author__ = 'bejar'

from netCDF4 import Dataset
import numpy as np
from Wind.Config.Paths import wind_data_path, wind_NREL_data_path

if __name__ == '__main__':
    lsites = []
    for dfile in range(126692):
        nc_fid = Dataset(f"{wind_NREL_data_path}/{dfile//500}/{dfile}.nc", 'r')
        lsites.append([dfile, nc_fid.getncattr('latitude'), nc_fid.getncattr('longitude')])
    vcoords = np.array(sorted(lsites, key=lambda x : x[0]))
    np.save(f"{wind_data_path}/Coords.npy", vcoords[:,1:])