"""
.. module:: GenerateCoordinatesFile

GenerateCoordinatesFile
*************

:Description: GenerateCoordinatesFile

    Extracts the longitude and latitude from the files and saves in the file Coords.npy

    Coords.py is a numpy array with longitude,latitude, the row corresponds with the site

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 10:38 

"""

__author__ = 'bejar'

from netCDF4 import Dataset
import numpy as np
from Wind.Config.Paths import wind_data_path, wind_NREL_data_path
from tqdm import tqdm

if __name__ == '__main__':
    lsites = []
    for dfile in tqdm(range(126692)):
        nc_fid = Dataset(f"{wind_NREL_data_path}/{dfile//500}/{dfile}.nc", 'r')
        lsites.append([dfile, nc_fid.getncattr('longitude'), nc_fid.getncattr('latitude')])
    vcoords = np.array(sorted(lsites, key=lambda x : x[0]))
    np.save(f"{wind_data_path}/Coords.npy", vcoords[:,1:])