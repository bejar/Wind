"""
.. module:: Test

Test
*************

:Description: Test

    

:Authors: bejar
    

:Version: 

:Created on: 07/06/2017 9:52 

"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import glob
from Wind.Config import wind_data, wind_data_ext, wind_path
from Wind.Maps.Util import MapThis
import os
import time

__author__ = 'bejar'


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''

    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr, \
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def plot_something():
    beg = 0
    end = 500
    data = []
    for i in range(2, 6):
        nc_fid = Dataset("/home/bejar/storage/Data/Wind/0/%d.nc" % i, 'r')
        data.append(nc_fid.variables['wind_direction'][beg:end])
    # nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

    fig = plt.figure(figsize=(10, 16), dpi=100)
    axes = fig.add_subplot(1, 1, 1)
    for d in data:
        axes.plot(range(end - beg), d)
    plt.show()


def explore_files(dir, ds):
    for v in os.listdir(dir + '/' + ds):
        yield dir + '/' + ds + '/' + v


def map_all():
    lds = [v for v in sorted(os.listdir(wind_data_ext)) if v[0] in '0123456789']
    print(lds)
    for ds in lds:
        print(ds)
        lcoords = []
        lfnames = []
        for f in explore_files(wind_data_ext, ds):
            data = Dataset(f, 'r')
            lcoords.append([data.latitude, data.longitude])
            lfnames.append(f)

        MapThis(lcoords, ds, lfnames)

if __name__ == '__main__':

    # lds = [v for v in sorted(os.listdir(wind_data_ext)) if v[0] in '0123456789']
    # wfile = open(wind_path + '/Results/Coords.csv', 'w')
    # for ds in lds:
    #     for f in explore_files(wind_data_ext, ds):
    #         data = Dataset(f, 'r')
    #         wfile.write('%f, %f, %s\n' % (data.latitude, data.longitude, f))
    #     wfile.flush()
    # 
    # wfile.close()

    itime = time.time()
    nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/0/0.nc", 'r')
    print(time.time() - itime)
    print(type(nc_fid.variables['wind_speed']))
    itime = time.time()
    v = np.array(nc_fid.variables['wind_speed'])
    print(time.time() - itime)
    print(type(v))
    print(v.shape)
