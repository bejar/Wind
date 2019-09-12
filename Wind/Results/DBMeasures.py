"""
.. module:: DBMeasures

DBMeasures
*************

:Description: DBMeasures

    Measures of time series characteristics

:Authors: bejar
    

:Version: 

:Created on: 03/09/2019 13:09 

"""
import numpy as np

from Wind.Config.Paths import wind_data_path
from Wind.ErrorMeasure import ErrorMeasure
import os
import warnings
from Wind.Util.Maps import create_mapbox_plot, create_plot, create_plot_best
import pandas as pd

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo = False
else:
    _has_mongo = True

__author__ = 'bejar'

scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"], \
      [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"], \
      [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [1, "rgb(255, 0, 0)"]

sclbi = [0, "rgb(0,255,255)"], [0.49999, "rgb(0, 0, 255)"], [0.5, "rgb(0, 0, 0)"], [0.50001, "rgb(255, 0, 0)"], [1,
                                                                                                                 "rgb(255, 255, 0)"]
sclbi = [0, "rgb(0,0,255)"], [0.49999, "rgb(0, 255, 255)"], [0.5, "rgb(0, 0, 0)"], [0.50001, "rgb(255, 255, 0)"], [1,"rgb(255, 0, 0)"]

sclbest = [0, "rgb(255,0,0)"], [0.1, "rgb(0,0,255)"], [0.2, "rgb(0,255,0)"], [0.3, "rgb(255,0,0)"], [0.4, "rgb(255,255,0)"], [0.5, "rgb(255,0,255)"], [0.4, "rgb(0,255,255)"]


class DBMeasures:
    """
    Access to results in a mongoDB and different plots and analysis
    """
    connection = None
    client = None
    user = None
    password = None
    db = None
    col = None
    coords = None
    osel = None
    selection = None

    variables = None
    measures = None

    exp_measures = None
    selection = None
    osel = None

    def __init__(self, conn=mongoconnection, test=''):
        """
        configures the DB
        :param DB:
        """
        if not _has_mongo:
            raise ("No pymongo library available")

        self.connection = conn
        self.client = MongoClient(conn.server)
        self.db = self.client[conn.db]
        if conn.passwd is not None:
            self.db.authenticate(conn.user, password=conn.passwd)
        self.col = self.db[conn.col + test]
        if os.path.isfile(f'{wind_data_path}/Coords.npy'):
            self.coords = np.load(wind_data_path + '/Coords.npy')
        else:
            warnings.warn('No coordinates file found, maps will not be available')
        self.exp_measures={}

    def retrieve_measures(self):
        """
        Retrieves measures from the DB for a query

        :param query:
        :return:
        """
        lexp = self.col.find({'experiment':'measures'})

        for exp in lexp:
            self.exp_measures[int(exp['site'].split('-')[1])] = exp['result']
        self.variables= list(exp['result'].keys())
        self.measures = list(exp['result'][self.variables[0]].keys())
        self.osel = list(self.exp_measures.keys())
        self.selection = self.osel

    def sample(self, percentage):
        """
        Picks a random sample from the measures

        :return:
        """
        if not self.exp_measures:
            raise NameError("No results yet retrieved")
        if 0 < percentage <= 1:
            sel = np.random.choice(self.osel,
                                   int(len(self.osel) * percentage), replace=False)
            self.selection = sel
        else:
            raise NameError("percentage must be in range (0-1]")

    def reset_selection(self):
        """
        Resets the selection of the results to show all the data
        :return:
        """
        self.selection = self.osel

    def plot_map(self,  notebook=False, cmap=scl, var='wind_speed', measure='SpecEnt', figsize=(1200, 800),mksize=2):
        """
        generates an html map with the results

        :param summary: Type of summary function
        :param notebook: If it is for a jupyter notebook
        :param cmap: colormap to apply
        :param mapbox: if it is a mapbix plot (needs access token to matplot)
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_measures:
            raise NameError("No measures yet retrieved")
        if self.coords is None:
            raise NameError("No coordinates file available")
        if var not in self.variables:
            raise NameError("Variable unknown")
        if measure not in self.measures:
            raise NameError("Measure unknown")

        title = f'{var} - {measure}'

        val = np.zeros(len(self.selection))
        for i, site in enumerate(self.selection):
            val[i] = self.exp_measures[site][var][measure]

        site_coords = self.selection

        testdf = pd.DataFrame({'Lon':self.coords[site_coords, 0],
                               'Lat': self.coords[site_coords, 1],
                               'Val': val,
                               'Site': self.selection}
                                    )

        create_plot(testdf, f"{title}", notebook=notebook, tick=10, cmap=cmap, figsize=figsize, mksize=mksize)

    def extract_measure(self, var='wind_speed', measure='SpecEnt'):
        """
        Returns the values of the measures for a variable with the site number
        :param var:
        :param measure:
        :return:
        """
        if not self.exp_measures:
            raise NameError("No measures yet retrieved")
        if self.coords is None:
            raise NameError("No coordinates file available")
        if var not in self.variables:
            raise NameError("Variable unknown")
        if measure not in self.measures:
            raise NameError("Measure unknown")

        val = np.zeros((len(self.selection), 2))
        for i, site in enumerate(self.selection):
            val[i,0] = site
            val[i,1] = self.exp_measures[site][var][measure]

        return val

    def extract_measure_sites(self, sites, var='wind_speed', measure='SpecEnt'):
        """
        Returns the values of the measures for a variable with the site number
        :param var:
        :param measure:
        :return:
        """
        if not self.exp_measures:
            raise NameError("No measures yet retrieved")
        if self.coords is None:
            raise NameError("No coordinates file available")
        if var not in self.variables:
            raise NameError("Variable unknown")
        if measure not in self.measures:
            raise NameError("Measure unknown")

        val = np.zeros((len(sites), 2))
        for i, site in enumerate(np.array(sites,dtype=int)):
            val[i,0] = site
            val[i,1] = self.exp_measures[site][var][measure]

        return val


if __name__ == '__main__':
    from Wind.Private.DBConfig import mongolocal, mongolocaltest

    results = DBMeasures(conn=mongolocaltest)

    results.retrieve_measures()
    print(results.variables)
    print(results.measures)
    results.sample(0.1)
    for var in results.variables:
        results.plot_map(var=var, measure='Stab1w')
