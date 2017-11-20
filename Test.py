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
        nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/0/%d.nc" % i, 'r')
        data.append(nc_fid.variables['wind_speed'][beg:end])
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
    # nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/0/0.nc", 'r')
    # ncdump(nc_fid)

    # lds = [v for v in sorted(os.listdir(wind_data_ext)) if v[0] in '0123456789']
    # wfile = open(wind_path + '/Results/Coords.csv', 'w')
    # for ds in lds:
    #     for f in explore_files(wind_data_ext, ds):
    #         data = Dataset(f, 'r')
    #         wfile.write('%f, %f, %s\n' % (data.latitude, data.longitude, f))
    #     wfile.flush()
    # 
    # wfile.close()

    # itime = time.time()
    # nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/0/0.nc", 'r')
    # print(time.time() - itime)
    # print(type(nc_fid.variables['wind_speed']))
    # itime = time.time()
    # # z = np.zeros((nc_fid.variables['wind_speed'].shape[0],2), dtype='float32')
    #
    # z = np.stack((np.array(nc_fid.variables['wind_speed']), np.array(nc_fid.variables['power'])), axis=1)
    # # v = np.array(nc_fid.variables['wind_speed'])
    # print(z.shape, z[0, 1])
    # print(time.time() - itime)
    # print(type(z))
    # print(z)
    # print(z.shape)


    wfiles = ['90/45142', '90/45143','90/45229','90/45230']
    vars = ['wind_speed', 'density', 'temperature', 'pressure']
    mdata = {}
    for wf in wfiles:
        print "/home/bejar/storage/Data/Wind/files/%s.nc" % wf
        nc_fid = Dataset("/home/bejar/storage/Data/Wind/files/%s.nc" % wf, 'r')
        ldata = []
        for v in vars:
            data = nc_fid.variables[v]
            print(data.shape)

            end = data.shape[0]
            step = 3
            length = int(end/step)
            print(length)
            data30 = np.zeros((length))

            for i in range(0, end, step):
                data30[i/step] = np.sum(data[i: i+step])/step

            ldata.append((data30))

        data30 = np.stack(ldata, axis=1)

        print(data30.shape)
        mdata[wf.replace('/', '-')] = data30
        # np.save('/home/bejar/wind%s.npy' % (wf.replace('/', '-')), data30)
    print(mdata)
    np.savez_compressed('/home/bejar/Wind.npz', **mdata)


    # fig = plt.figure(figsize=(10, 16), dpi=100)
    # axes = fig.add_subplot(1, 1, 1)
    # axes.plot(range(length), data30)
    # plt.show()
