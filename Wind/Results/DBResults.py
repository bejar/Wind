"""
.. module:: DBResults

DBResults
*************

:Description: DBResults

    Access to the results in a mongoDB

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 15:09 

"""
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import seaborn as sb

from Wind.Config.Paths import wind_data_path

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

# Plotly colorscales
#
#[‘Blackbody’, ‘Bluered’, ‘Blues’, ‘Earth’, ‘Electric’, ‘Greens’, ‘Greys’, ‘Hot’, ‘Jet’,
#‘Picnic’, ‘Portland’, ‘Rainbow’, ‘RdBu’, ‘Reds’, ‘Viridis’, ‘YlGnBu’, ‘YlOrRd’]


def create_plot(df, title, notebook=False, tick=10, cmap=scl):
    """
    Creates an HTML file with the map
    """
    data = [dict(
        lat=df['Lat'],
        lon=df['Lon'],
        text=df['Val'].astype(str) + '-[' + df['Site'].astype(str) + ']',
        marker=dict(
            color=df['Val'],
            colorscale=cmap,
            reversescale=True,
            opacity=0.7,
            size=2,
            colorbar=dict(
                thickness=10,
                titleside="right",
                outlinecolor="rgba(68, 68, 68, 0)",
                ticks="outside",
                ticklen=3,
                showticksuffix="last",
                ticksuffix=" ",
                dtick=tick/10
            ),
        ),
        type='scattergeo'
    )]

    layout = dict(
        geo=dict(
            scope='north america',
            showland=True,
            landcolor="rgb(212, 212, 212)",
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            showsubunits=True,
            showcountries=True,
            resolution=50,
            projection=dict(
                # type='conic conformal',
                type='mercator',
                rotation=dict(
                    lon=-100
                )
            ),
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[-140.0, -55.0],
                dtick=5
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[20.0, 60.0],
                dtick=5
            )
        ),
        width=1200,
        height=800,
        title=title,
    )
    fig = {'data': data, 'layout': layout}

    if notebook:
        py.iplot(fig)
    else:
        py.plot(fig, filename='./' + title + '.html')

class DBResults:
    """
    Access to results in a mongoDB
    """
    connection = None
    client = None
    user = None
    password = None
    db = None
    col = None
    coords = None
    query = None

    # Stores the results for test and validation as numpy arrays
    exp_result = {}

    def __init__(self, conn=mongoconnection, test=""):
        """
        configures the DB
        :param DB:
        """
        self.connection = conn
        self.client = MongoClient(conn.server)
        self.db = self.client[conn.db]
        if conn.passwd is not None:
            self.db.authenticate(conn.user, password=conn.passwd)
        self.col = self.db[conn.col + test]
        self.coords = np.load(wind_data_path + '/Coords.npy')

    def size(self):
        """
        Returns the number of results obtained

        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")

        return self.exp_result['sites'].shape[0]

    def retrieve_results(self, query):
        """
        Retrieves results from the DB for a query

        :param query:
        :return:
        """
        self.query = query

        lexp = self.col.find(query)
        ldata = []
        for exp in lexp:
            # To maintain backwards compatibility
            if 'result' in exp:
                data = np.array(exp['result'])
            elif 'results' in exp:
                data = np.array(exp['results'])

            # gets the number of the site and the columns with the results
            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, 1], data[:, 2]))
        ldata = sorted(ldata, key=lambda x: x[0])
        self.exp_result['sites'] = np.array([v[0] for v in ldata])
        self.exp_result['test'] = np.array([v[1] for v in ldata])
        self.exp_result['validation'] = np.array([v[2] for v in ldata])

    def select_best_worst_sum_accuracy(self, summary='sum', size=100):
        """
        Selects a number of experiment with the best and worst summarized accuracy from the test

        by default uses the sum

        :param exp:
        :param mode:
        :param upper:
        :param lower:
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        dtype = [('site', int), ('summary', float)]

        if summary == 'sum':
            dsum = np.sum(self.exp_result['test'], axis=1)
        else:
            dsum = self.exp_result['test'][:,0]

        mdata = np.sort(np.array(list(zip(range(self.exp_result['test'].shape[0]),dsum)),
                                 dtype=dtype),
                        order='summary')

        sel = [v[0] for v in mdata[:size]]
        sel.extend([v[0] for v in mdata[-size:]])

        self.exp_result['sites'] = self.exp_result['sites'][sel]
        self.exp_result['test'] = self.exp_result['test'][sel]
        self.exp_result['validation'] = self.exp_result['validation'][sel]

    def sample(self, percentage):
        """
        Picks a random sample from the results

        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        if 0 < percentage <= 1:
            sel = np.random.choice(self.exp_result['sites'].shape[0],
                                   int(self.exp_result['sites'].shape[0] * percentage), replace=False)
            self.exp_result['sites'] = self.exp_result['sites'][sel]
            self.exp_result['test'] = self.exp_result['test'][sel]
            self.exp_result['validation'] = self.exp_result['validation'][sel]
        else:
            raise NameError("percentage must be in range (0,1]")

    def plot_map(self, summary='sum', notebook=False, cmap=scl):
        """
        generates an html map with the results
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")

        else:
            if 'experiment' in self.query:
                title = self.query['experiment']
            else:
                title = 'NonSpecific'

            if summary == 'sum':
                sumtest = np.sum(self.exp_result['test'], axis=1)
                sumval = np.sum(self.exp_result['validation'], axis=1)
                extra = [10, 1]
            else:
                sumtest = self.exp_result['test'][:,0]
                sumval = self.exp_result['validation'][:,0]
                extra = [1,0]

            create_plot(
                pd.DataFrame({'Lon': np.append(self.coords[self.exp_result['sites'], 0], [0, 0]),
                              'Lat': np.append(self.coords[self.exp_result['sites'], 1], [0, 0]),
                              'Val': np.append(sumtest, extra),
                              'Site': np.append(self.exp_result['sites'], [0, 0])}),
                f"{title}-test", notebook=notebook, tick=extra[0], cmap=cmap
            )
            create_plot(
                pd.DataFrame({'Lon': np.append(self.coords[self.exp_result['sites'], 0], [0, 0]),
                              'Lat': np.append(self.coords[self.exp_result['sites'], 1], [0, 0]),
                              'Val': np.append(sumval, extra),
                              'Site': np.append(self.exp_result['sites'], [0, 0])}),
                f"{title}-validation", notebook=notebook, tick=extra[0], cmap=cmap
            )


    def plot_kde(self, summary='sum'):
        """
        Generates a kernel density estimation plot of the results

        :param summary:
        :return:
        """


    # ---------------------------------



    def find_exp(self, query):
        """
        Returns all the experiments in the DB that match the query

        :param query:
        :return:
        """
        return self.col.find(query)

    def count_exp(self, query):
        """
        Counts how many experiments in the DB match the query

        :param query:
        :return:
        """
        return self.col.count(query)

    def sel_result(self, query, ncol):
        """
        Selects from a list of configurations with results the result in the column 1 (test) or 2 (validation)

        :param query:
        :param lexp:
        :param ncol:
        :return:
        """
        lexp = self.find_exp(query)
        ldata = []
        for exp in lexp:

            # To maintain backwards compatibility
            if 'result' in exp:
                data = np.array(exp['result'])
            elif 'results' in exp:
                data = np.array(exp['results'])

            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, ncol]))
        ldata = sorted(ldata, key=lambda x: x[0])

        return np.array([v[0] for v in ldata]), np.array([v[1] for v in ldata])


if __name__ == '__main__':

    query = {'status': 'done', "experiment": 'Persistence', "site": {"$regex": "."}}
    results = DBResults()
    results.retrieve_results(query)
    if results.size() > 0:
        print(results.select_best_worst_sum_accuracy(summary=None))
