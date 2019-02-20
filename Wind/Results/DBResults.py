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
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, ttest_rel,f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from Wind.Config.Paths import wind_data_path
from statsmodels.genmod.generalized_linear_model import GLM

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection, mapbox_token
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
# Plotly colorscales
#
# [‘Blackbody’, ‘Bluered’, ‘Blues’, ‘Earth’, ‘Electric’, ‘Greens’, ‘Greys’, ‘Hot’, ‘Jet’,
# ‘Picnic’, ‘Portland’, ‘Rainbow’, ‘RdBu’, ‘Reds’, ‘Viridis’, ‘YlGnBu’, ‘YlOrRd’]

def create_mapbox_plot(df, title, notebook=False, tick=10, cmap=scl, figsize=(1200, 800)):
    """
    Maps using mapbox
    :return:
    """
    data = [
        go.Scattermapbox(
            lat=df['Lat'],
            lon=df['Lon'],
            mode='markers',
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
                    dtick=tick / 10
                ))
        )
    ]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_token,
            bearing=0,
            center=dict(
                lat=35,
                lon=-100
            ),
            pitch=0,
            zoom=3
        ), width=figsize[0],
        height=figsize[1],
        title=title,

    )

    fig = dict(data=data, layout=layout)

    if notebook:
        py.iplot(fig)
    else:
        py.plot(fig, filename=f"./{title}.html")


def create_plot(df, title, notebook=False, tick=10, cmap=scl, figsize=(1200, 800)):
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
                dtick=tick / 10
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
                range=[-126.0, -65.0],
                dtick=5
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[23.0, 52.0],
                dtick=5
            )
        ),
        width=figsize[0],
        height=figsize[1],
        title=title,
    )
    fig = {'data': data, 'layout': layout}

    if notebook:
        py.iplot(fig)
    else:
        py.plot(fig, filename=f"./{title}.html")


def create_plot_best(df, title, labels, notebook=False, image=False, tick=10, cmap=scl, figsize=(1200, 800)):
    """
    Creates an HTML file with the map
    """
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    data = dict(
        mode='markers',
        type='scattergeo'
    )

    ldata = []
    for i in range(len(labels)):
        ndata = data.copy()
        ndata['lat'] = df[df['Val']==i]['Lat']
        ndata['lon'] = df[df['Val']==i]['Lon']
        ndata['marker'] = dict(size=1,color=i)
        ndata['name'] = labels[i]
        ldata.append(ndata)
       

    layout = dict(
       legend=dict(font=dict(size=20)), 
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
                range=[-126.0, -65.0],
                dtick=5
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[23.0, 52.0],
                dtick=5
            )
        ),
        width=figsize[0],
        height=figsize[1],
        title=title,
    )
    fig = {'data': ldata, 'layout': layout}

    if notebook:
        py.iplot(fig)
    else:
        py.plot(fig, filename=f"./{title}.html", image='svg')


class DBResults:
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
    query = None
    lquery = None
    osel = None
    selection = None

    # Stores the results for test and validation as numpy arrays
    exp_result = {}
    exp_result2 = {}
    exp_lresults = []
    exp_df = None

    def __init__(self, conn=mongoconnection, test=""):
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
        self.coords = np.load(wind_data_path + '/Coords.npy')

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
            if len(data) == 0: #Hack to avoid sites with no results
                data = np.zeros((12,3))
            # gets the number of the site and the columns with the results
            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, 1], data[:, 2]))
        if len(ldata) == 0:
            raise NameError('No results retrieved with query')
        ldata = sorted(ldata, key=lambda x: x[0])
        self.exp_result['sites'] = np.array([v[0] for v in ldata])
        self.exp_result['test'] = np.array([v[1] for v in ldata])
        self.exp_result['validation'] = np.array([v[2] for v in ldata])
        self.osel = list(range(self.exp_result['sites'].shape[0]))
        self.selection = list(range(self.exp_result['sites'].shape[0]))

    def retrieve_results_compare(self, query1, query2):
        """
        Retrieves two sets of results to compare them

        :param query1:
        :param query2:
        :return:
        """
        # Result 1
        self.query = query1
        lexp = self.col.find(query1)


        ldata = []
        for exp in lexp:
            # To maintain backwards compatibility
            if 'result' in exp:
                data = np.array(exp['result'])
            elif 'results' in exp:
                data = np.array(exp['results'])

            # gets the number of the site and the columns with the results
            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, 1], data[:, 2]))
        if len(ldata) == 0:
            raise NameError('No results retrieved with first query')

        ldata = sorted(ldata, key=lambda x: x[0])
        self.exp_result['sites'] = np.array([v[0] for v in ldata])
        self.exp_result['test'] = np.array([v[1] for v in ldata])
        self.exp_result['validation'] = np.array([v[2] for v in ldata])
        self.osel = list(range(self.exp_result['sites'].shape[0]))
        self.selection = list(range(self.exp_result['sites'].shape[0]))

        # Result 2
        self.query2 = query2
        lexp = self.col.find(query2)

        ldata = []
        for exp in lexp:
            # To maintain backwards compatibility
            if 'result' in exp:
                data = np.array(exp['result'])
            elif 'results' in exp:
                data = np.array(exp['results'])

            # gets the number of the site and the columns with the results
            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, 1], data[:, 2]))

        if len(ldata) == 0:
            raise NameError('No results retrieved with second query')
        ldata = sorted(ldata, key=lambda x: x[0])
        # Check that all the sites in one result set are in the other
        s1 = set(self.exp_result['sites'])
        s2 = set([v[0] for v in ldata])
        if s1-s2:
            raise NameError('Results not comparable, sites do not match')
        else:
            self.exp_result2['test'] = np.array([v[1] for v in ldata])
            self.exp_result2['validation'] = np.array([v[2] for v in ldata])

    def retrieve_results_multiple(self, lquery):
        """
        Retrieves a list of sets of results to compare them (just for ANOVA for now)

        :param query1:
        :param query2:
        :return:
        """
        self.lquery = lquery
        for i, query in enumerate(lquery):
            lexp = self.col.find(query)
            exp_result = {}

            ldata = []
            for exp in lexp:
                # To maintain backwards compatibility
                if 'result' in exp:
                    data = np.array(exp['result'])
                elif 'results' in exp:
                    data = np.array(exp['results'])

                # gets the number of the site and the columns with the results
                ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, 1], data[:, 2]))
            if len(ldata) == 0:
                raise NameError(f'No results retrieved with {i}-th query')

            ldata = sorted(ldata, key=lambda x: x[0])

            # Check that the sites are compatible
            if self.exp_lresults:
                s1 = set(self.exp_lresults[0]['sites'])
                s2 = set([v[0] for v in ldata])
                if s1-s2:
                    raise NameError(f'Results not comparable, sites do not match for {i}')
                else:
                    exp_result['sites'] = np.array([v[0] for v in ldata])
                    exp_result['test'] = np.array([v[1] for v in ldata])
                    exp_result['validation'] = np.array([v[2] for v in ldata])
                    self.exp_lresults.append(exp_result)
            else:
                exp_result['sites'] = np.array([v[0] for v in ldata])
                exp_result['test'] = np.array([v[1] for v in ldata])
                exp_result['validation'] = np.array([v[2] for v in ldata])
                self.exp_lresults.append(exp_result)


        self.osel = list(range(self.exp_lresults[0]['sites'].shape[0]))
        self.selection = list(range(self.exp_lresults[0]['sites'].shape[0]))


    def retrieve_results_dataframe(self, query, data=('vars', 'ahead', 'lag'), arch=[], train=[]):
        """
        Retrieves the results from the query and builds a pandas dataframe with all the data

        :param query:
        :return:
        """
        lexp = self.col.find(query)
        lvars = ['hour', 'site', 'test', 'val']
        if data:
            lvars.extend(list(data))
        if arch:
            lvars.extend(arch)
        if train:
            lvars.extend(train)
        ddata = {}
        for var in lvars:
            ddata[var] = []

        for exp in lexp:
            # To maintain backwards compatibility
            if 'result' in exp:
                exdata = np.array(exp['result'])
            elif 'results' in exp:
                exdata = np.array(exp['results'])

            for i in range(exdata.shape[0]):
                lvals = [i+1]
                lvals.append(int(exp['data']['datanames'][0].split('-')[1]))
                lvals.append(exdata[i, 1])
                lvals.append(exdata[i, 2])

                for v in data:
                    lvals.append(str(exp['data'][v]))

                for v in arch:
                    lvals.append(str(exp['arch'][v]))

                for v in train:
                    lvals.append(str(exp['training'][v]))

                for var, val in zip(lvars, lvals):
                    ddata[var].append(val)

        self.exp_df = pd.DataFrame(ddata)

    def size(self):
        """
        Returns the number of results obtained

        :return:
        """
        if not self.exp_result and (self.exp_df is None):
            raise NameError("No results yet retrieved")

        if self.exp_result:
            return self.exp_result['sites'].shape[0]
        elif self.exp_lresults:
            return self.exp_lresults[0]['sites'].shape[0]
        else:
            return self.exp_df.shape[0]

    def selected_size(self):
        """
        Returns the number of results obtained

        :return:
        """
        if not self.exp_result and not self.exp_lresults:
            raise NameError("No results yet retrieved")

        return len(self.selection)

    def reset_selection(self):
        """
        Resets the selection of the results to show all the data
        :return:
        """
        self.selection = list(range(self.exp_result['sites'].shape[0]))

    def select_best_worst(self, summary='sum', size=100):
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

        if type(summary) == int:
            dsum = self.exp_result['test'][:, summary]
        elif summary == 'sum':
            dsum = np.sum(self.exp_result['test'], axis=1)
        else:
            dsum = self.exp_result['test'][:, 0]

        mdata = np.sort(np.array(list(zip(range(self.exp_result['test'].shape[0]), dsum)),
                                 dtype=dtype),
                        order='summary')

        sel = [v[0] for v in mdata[:size]]
        sel.extend([v[0] for v in mdata[-size:]])
        self.selection = sel

        # self.exp_result['sites'] = self.exp_result['sites'][sel]
        # self.exp_result['test'] = self.exp_result['test'][sel]
        # self.exp_result['validation'] = self.exp_result['validation'][sel]

    def select_geo(self, igeo, fgeo):
        """
        Selects the sites inside a pair of longitude/latitude coordinates
        :return:
        """
        ilon, ilat = igeo
        flon, flat = fgeo

        if ilat > flat:
            tmp = flat
            flat = ilat
            ilat = tmp

        if ilon > flon:
            tmp = flon
            flon = ilon
            ilon = tmp

        if not (-130 <= ilon <= -63) or not (-130 <= flon <= -63) or not (20 <= ilat <= 50) or not (20 <= flat <= 50):
            raise NameError("Coordinates outside range, use longitude in [-130,-63] and latitude in [20, 50]")

        self.selection = [i for i in self.osel if
                          (ilat <= self.coords[i][1] <= flat) and (ilon <= self.coords[i][0] <= flon)]

    def sample(self, percentage):
        """
        Picks a random sample from the results

        :return:
        """
        if not self.exp_result and not self.exp_lresults:
            raise NameError("No results yet retrieved")
        if 0 < percentage <= 1:
            sel = np.random.choice(self.osel,
                                   int(len(self.osel) * percentage), replace=False)
            self.selection = sel
        else:
            raise NameError("percentage must be in range (0-1]")

    def plot_map(self, summary='sum', notebook=False, cmap=scl, mapbox=False, dset=('val', 'test'), figsize=(800, 400)):
        """
        generates an html map with the results

        :param summary: Type of summary function
        :param notebook: If it is for a jupyter notebook
        :param cmap: colormap to apply
        :param mapbox: if it is a mapbix plot (needs access token to matplot)
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")

        if 'experiment' in self.query:
            title = self.query['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            extra = [1, 0]

        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            extra = [10, 0]

        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            extra = [1, 0]

        if 'test' in dset:
            testdf = pd.DataFrame({'Lon': np.append(self.coords[self.selection, 0], [0, 0]),
                                   'Lat': np.append(self.coords[self.selection, 1], [0, 0]),
                                   'Val': np.append(sumtest, extra),
                                   'Site': np.append(self.exp_result['sites'][self.selection], [0, 0])})
        if 'val' in dset:
            valdf = pd.DataFrame({'Lon': np.append(self.coords[self.selection, 0], [0, 0]),
                                  'Lat': np.append(self.coords[self.selection, 1], [0, 0]),
                                  'Val': np.append(sumval, extra),
                                  'Site': np.append(self.exp_result['sites'][self.selection], [0, 0])})

        if mapbox:
            if 'test' in dset:
                create_mapbox_plot(testdf,
                                   f"{title}-test", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                                   )
            if 'val' in dset:
                create_mapbox_plot(valdf,
                                   f"{title}-validation", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                                   )
        else:
            if 'test' in dset:
                create_plot(testdf,
                            f"{title}-test", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                            )
            if 'val' in dset:
                create_plot(valdf,
                            f"{title}-validation", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                            )

    def plot_map_compare(self, summary='sum', notebook=False, compare='diff', cmap=sclbi, mapbox=False,
                         dset=('val', 'test'), figsize=(800, 400)):
        """
        generates an html map with the results

        :param compare:
        :param summary: Type of summary function
        :param notebook: If it is for a jupyter notebook
        :param cmap: colormap to apply
        :param mapbox: if it is a mapbix plot (needs access token to matplot)
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")

        if 'experiment' in self.query:
            title = self.query['experiment'] + '-vs-' + self.query2['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            sumtest2 = self.exp_result2['test'][self.selection, summary]
            sumval2 = self.exp_result2['validation'][self.selection, summary]
        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            sumtest2 = np.sum(self.exp_result2['test'][self.selection], axis=1)
            sumval2 = np.sum(self.exp_result2['validation'][self.selection], axis=1)
        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            sumtest2 = self.exp_result2['test'][self.selection, 0]
            sumval2 = self.exp_result2['validation'][self.selection, 0]

        if compare == 'diff':
            difftest = sumtest - sumtest2
            diffval = sumval - sumval2
            mx = max(np.abs(max(np.max(difftest), np.max(diffval))),
                     np.abs(min(np.min(difftest), np.min(diffval))))
            extra = [mx, -mx]
        else:
            difftest = sumtest - sumtest2
            diffval = sumval - sumval2
            mx = max(np.abs(max(np.max(difftest), np.max(diffval))),
                     np.abs(min(np.min(difftest), np.min(diffval))))
            extra = [mx, -mx]

        if 'test' in dset:
            testdf = pd.DataFrame({'Lon': np.append(self.coords[self.selection, 0], [0, 0]),
                                   'Lat': np.append(self.coords[self.selection, 1], [0, 0]),
                                   'Val': np.append(difftest, extra),
                                   'Site': np.append(self.exp_result['sites'][self.selection], [0, 0])})
        if 'val' in dset:
            valdf = pd.DataFrame({'Lon': np.append(self.coords[self.selection, 0], [0, 0]),
                                  'Lat': np.append(self.coords[self.selection, 1], [0, 0]),
                                  'Val': np.append(diffval, extra),
                                  'Site': np.append(self.exp_result['sites'][self.selection], [0, 0])})

        if mapbox:
            if 'test' in dset:
                create_mapbox_plot(testdf,
                                   f"{title}-test", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                                   )
            if 'val' in dset:
                create_mapbox_plot(valdf,
                                   f"{title}-validation", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                                   )
        else:
            if 'test' in dset:
                create_plot(testdf,
                            f"{title}-test", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                            )
            if 'val' in dset:
                create_plot(valdf,
                            f"{title}-validation", notebook=notebook, tick=extra[0], cmap=cmap, figsize=figsize
                            )

    def plot_map_multiple(self, summary='sum', notebook=False, labels=None,
                         dset=('val', 'test'), figsize=(800, 400), image=False):
        """
        generates an html map with the results

        :param summary: Type of summary function
        :param notebook: If it is for a jupyter notebook
        :param cmap: colormap to apply
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_lresults:
            raise NameError("No results yet retrieved")

        # if 'experiment' in self.query:
        #     title = self.query['experiment'] + '-vs-' + self.query2['experiment']
        # else:
        #     title = 'NonSpecific'
        title = 'NonSpecific'
        msumtest = []
        msumval = []
        #labels = [l['experiment'] for l in self.lquery]
        for exp_result in self.exp_lresults:
            if type(summary) == int:
                msumtest.append(exp_result['test'][self.selection, summary])
                msumval.append(exp_result['validation'][self.selection, summary])
            elif summary == 'sum':
                msumtest.append(np.sum(exp_result['test'][self.selection], axis=1))
                msumval.append(np.sum(exp_result['validation'][self.selection], axis=1))
            else:
                msumtest.append(exp_result['test'][self.selection, 0])
                msumval.append(exp_result['validation'][self.selection, 0])

        msumtest = np.stack(msumtest, axis=1)
        msumtval = np.stack(msumval, axis=1)

        difftest = np.argmax(msumtest,axis=1)
        diffval = np.argmax(msumtval,axis=1)


        if 'test' in dset:
            testdf = pd.DataFrame({'Lon': self.coords[self.selection, 0],
                                   'Lat': self.coords[self.selection, 1],
                                   'Val': difftest,
                                   'Site': self.exp_lresults[0]['sites'][self.selection]})
        if 'val' in dset:
            valdf = pd.DataFrame({'Lon': self.coords[self.selection, 0],
                                  'Lat': self.coords[self.selection, 1],
                                  'Val': diffval,
                                  'Site': self.exp_lresults[0]['sites'][self.selection]})

        if 'test' in dset:
            create_plot_best(testdf,
                        f"{title}-test", notebook=notebook, figsize=figsize, labels=labels
                        )
        if 'val' in dset:
            create_plot_best(valdf,
                        f"{title}-validation", notebook=notebook, figsize=figsize, labels=labels
                        )



    def plot_distplot(self, summary='sum', seaborn=True, notebook=False, dset=('val', 'test'), figsize=(800, 400)):
        """
        Generates a distplot of the results

        :param summary: Type of summary function
        :param seaborn: Uee seaborn instead of plotly
        :param notebook: If it is for a jupyter notebook
        :param cmap: colormap to apply
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            extra = [1, 0]
        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            extra = [10, 0]
        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            extra = [10, 1]

        data = []
        labels = []
        if 'test' in dset:
            data.append(np.append(sumtest, extra))
            labels.append('test')
        if 'val' in dset:
            data.append(np.append(sumval, extra))
            labels.append('val')


        if seaborn:
            plt.figure(figsize=figsize)
            for v,l in zip(data, labels):
                sns.distplot(v, label=l, kde=True, norm_hist=True)
                plt.legend(labels=labels, title=title)
        else:
            fig = ff.create_distplot(data, labels, bin_size=.05)
            fig.layout.width = figsize[0]
            fig.layout.height = figsize[1]

            if notebook:
                py.iplot(fig)
            else:
                py.iplot(fig, filename=f"./{title}-distplot.html")

    def plot_distplot_compare(self, summary='sum', compare='diff', seaborn=True, notebook=False, dset=('val', 'test'),
                              figsize=(800, 400)):
        """
        Generates a distplot for the comparison of the results results

        :param compare:
        :param summary: Type of summary function
        :param seaborn: Uee seaborn instead of plotly
        :param notebook: If it is for a jupyter notebook
        :param dset: If plots for validation/test set (must be a list)
        :return:
        """
        if not self.exp_result or not self.exp_result2:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment'] + '-vs-' + self.query2['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            sumtest2 = self.exp_result2['test'][self.selection, summary]
            sumval2 = self.exp_result2['validation'][self.selection, summary]

        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            sumtest2 = np.sum(self.exp_result2['test'][self.selection], axis=1)
            sumval2 = np.sum(self.exp_result2['validation'][self.selection], axis=1)

        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            sumtest2 = self.exp_result2['test'][self.selection, 0]
            sumval2 = self.exp_result2['validation'][self.selection, 0]

        data = []
        labels = []
        if compare == 'diff':
            difftest = sumtest - sumtest2
            diffval = sumval - sumval2
            extra = [max(np.max(difftest), np.max(diffval)), min(np.min(difftest), np.min(diffval))]
            if 'test' in dset:
                data.append(np.append(difftest, extra))
                labels.append('test ' + title)
            if 'val' in dset:
                data.append(np.append(diffval, extra))
                labels.append('val ' + title)
        else:

            extra = [max(np.max(sumtest), np.max(sumval)), min(np.min(sumtest), np.min(sumval))]
            if 'test' in dset:
                data.append(np.append(sumtest, extra))
                data.append(np.append(sumtest2, extra))
                labels.append('test ' + self.query['experiment'])
                labels.append('test ' + self.query2['experiment'])
            if 'val' in dset:
                data.append(np.append(sumval, extra))
                data.append(np.append(sumval2, extra))
                labels.append('val ' + self.query['experiment'])
                labels.append('val ' + self.query2['experiment'])

        if seaborn:
            plt.figure(figsize=figsize)
            for v,l in zip(data, labels):
                sns.distplot(v, label=l, kde=True, norm_hist=True)
                plt.legend(labels=labels, title=title)
        else:
            fig = ff.create_distplot(data, labels, bin_size=.05)
            fig.layout.width = figsize[0]
            fig.layout.height = figsize[1]
            if notebook:
                py.iplot(fig)
            else:
                py.iplot(fig, filename=f"./{title}-distplot.html")

    def plot_densplot(self, plot='kde', glm=False, summary='sum', figsize=(8, 8)):
        """
        Generates a density plot comparing test and validation

        :param summary: Type of summary function
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]

        data = pd.DataFrame({'test':sumtest, 'validation':sumval})
        plt.figure(figsize=figsize)
        if plot == 'kde':
            sns.kdeplot(data['test'],data['validation'], shade=True, n_levels=10, cbar=True, shade_lowest=False)
        elif plot == 'regression':
            sns.regplot('test','validation', data=data,truncate=True, line_kws={'color':'red', 'linewidth':2,'linestyle':'--'})
        else:
            sns.scatterplot('test','validation', data=data)
        if glm:
            model = GLM.from_formula('test ~ validation', data)
            result = model.fit()
            print(result.summary())
            

    def plot_densplot_compare(self, plot='kde', glm=False, summary='sum', figsize=(8, 8)):
        """
        Generates a density plot comparing test and validation

        :param summary: Type of summary function
        :return:
        """
        if not self.exp_result or not self.exp_result2:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment'] + '-vs-' + self.query2['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            sumtest2 = self.exp_result2['test'][self.selection, summary]
            sumval2 = self.exp_result2['validation'][self.selection, summary]

        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            sumtest2 = np.sum(self.exp_result2['test'][self.selection], axis=1)
            sumval2 = np.sum(self.exp_result2['validation'][self.selection], axis=1)
        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            sumtest2 = self.exp_result2['test'][self.selection, 0]
            sumval2 = self.exp_result2['validation'][self.selection, 0]

        data1 = pd.DataFrame({'test':sumtest, 'validation':sumval})
        data2 = pd.DataFrame({'test':sumtest2, 'validation':sumval2})

        plt.figure(figsize=figsize)
        if plot == 'kde':
            sns.kdeplot(data1['test'],data1['validation'], cmap="Reds", n_levels=10, cbar=True, shade_lowest=False)
            sns.kdeplot(data2['test'],data2['validation'], cmap="Blues", n_levels=10, cbar=True, shade_lowest=False)
        elif plot == 'regression':
            sns.regplot('test','validation', data=data1,truncate=True, line_kws={'color':'red', 'linewidth':2,'linestyle':'--'})
            sns.regplot('test','validation', data=data2,truncate=True, line_kws={'color':'blue', 'linewidth':2,'linestyle':'--'})
        else:
            sns.scatterplot('test','validation', data=data1)
            sns.scatterplot('test','validation', data=data2)
        if glm:
            model = GLM.from_formula('test ~ validation', data1)
            result = model.fit()
            print(result.summary())
            model = GLM.from_formula('test ~ validation', data2)
            result = model.fit()
            print(result.summary())

    def plot_hours_boxplot(self, dset=('val', 'test'), figsize=(8, 4)):
        """
        Plots the accuracy for each hour in a boxplot

        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment']
        else:
            title = 'NonSpecific'

        data = np.array([])
        hour = np.array([])
        exp = np.array([])
        if 'test' in dset:

            for i in range(self.exp_result['test'].shape[1]):
                data = np.append(data, self.exp_result['test'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['test']*len(self.selection)))

        if 'val' in dset:
            for i in range(self.exp_result['validation'].shape[1]):
                data = np.append(data, self.exp_result['validation'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['val']*len(self.selection)))

        df = pd.DataFrame({'hour':hour, 'acc':data, title:exp})
        plt.figure(figsize=figsize, dpi=100)
        sns.boxplot(x='hour', y='acc', hue=title,data=df)

        plt.show()

    def plot_hours_boxplot_compare(self, dset=('val', 'test'), figsize=(16, 4)):
        """
        Plots the accuracy for each hour in a boxplot

        :return:
        """
        if not self.exp_result or not self.exp_result2:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment'] + '-vs-' + self.query2['experiment']
        else:
            title = 'NonSpecific'


        data = np.array([])
        hour = np.array([])
        exp = np.array([])
        if 'test' in dset:

            for i in range(self.exp_result['test'].shape[1]):
                data = np.append(data, self.exp_result['test'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['test-'+self.query['experiment']]*len(self.selection)))

            for i in range(self.exp_result2['test'].shape[1]):
                data = np.append(data, self.exp_result2['test'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['test-'+self.query2['experiment']]*len(self.selection)))

        if 'val' in dset:
            for i in range(self.exp_result['validation'].shape[1]):
                data = np.append(data, self.exp_result['validation'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['val-'+self.query['experiment']]*len(self.selection)))

            for i in range(self.exp_result2['validation'].shape[1]):
                data = np.append(data, self.exp_result2['validation'][self.selection, i])
                hour = np.append(hour, np.array([int(i+1)]*len(self.selection)))
                exp = np.append(exp, np.array(['val-'+self.query2['experiment']]*len(self.selection)))

        df = pd.DataFrame({'hour':hour, 'acc':data, title:exp})
        plt.figure(figsize=figsize, dpi=100)
        sns.boxplot(x='hour', y='acc', hue=title,data=df)

        plt.show()


    def plot_2DKDEplot(self, summary='sum', notebook=False, dset=('val', 'test'), figsize=(800, 400)):
        """
        Plots a 2D KDE plot with seaborn
        It shows the summary accuracy density per longitude and latitude

        Work in progress

        :param summary:
        :param notebook:
        :param dset:
        :param figsize:
        :return:
        """
        if not self.exp_result:
            raise NameError("No results yet retrieved")
        if 'experiment' in self.query:
            title = self.query['experiment']
        else:
            title = 'NonSpecific'

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            extra = [1, 0]
        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            extra = [10, 0]
        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            extra = [10, 1]

        f, axes = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=True)

        sns.kdeplot(self.coords[self.selection, 0], sumval,
                    cmap="Reds", shade=True, shade_lowest=False, ax=axes.flat[0], cbar=True)

        sns.kdeplot(self.coords[self.selection, 1], sumval,
                    cmap="Reds", shade=True, shade_lowest=False, ax=axes.flat[1], cbar=True)

        plt.show()

    def compare_2samp(self, summary='sum',dset=('val', 'test')):
        """
        Computes a 2 sided kolmogorov-smirnov test for equality of distribution
        Computes a T-test for two related samples for identical average

        Given that the results are large a subsample should be used
        :return:
        """
        if not self.exp_result or not self.exp_result2:
            raise NameError("No results yet retrieved")

        if type(summary) == int:
            sumtest = self.exp_result['test'][self.selection, summary]
            sumval = self.exp_result['validation'][self.selection, summary]
            sumtest2 = self.exp_result2['test'][self.selection, summary]
            sumval2 = self.exp_result2['validation'][self.selection, summary]

        elif summary == 'sum':
            sumtest = np.sum(self.exp_result['test'][self.selection], axis=1)
            sumval = np.sum(self.exp_result['validation'][self.selection], axis=1)
            sumtest2 = np.sum(self.exp_result2['test'][self.selection], axis=1)
            sumval2 = np.sum(self.exp_result2['validation'][self.selection], axis=1)

        else:
            sumtest = self.exp_result['test'][self.selection, 0]
            sumval = self.exp_result['validation'][self.selection, 0]
            sumtest2 = self.exp_result2['test'][self.selection, 0]
            sumval2 = self.exp_result2['validation'][self.selection, 0]

        res = {}
        if 'test' in dset:
            res['test'] ={}
            res['test']['KS2samp'] = ks_2samp(sumtest, sumtest2)
            res['test']['T-test'] = ttest_rel(sumtest, sumtest2)
        if 'val' in dset:
            res['val']={}
            res['val']['KS2samp'] = ks_2samp(sumval, sumval2)
            res['val']['T-test'] = ttest_rel(sumval, sumval2)
        return res

    def compare_ANOVA(self, summary='sum',dset=('val', 'test'), labels=None):
        """
        Computes a 2 sided kolmogorov-smirnov test for equality of distribution

        Given that the results are large a subsample should be used
        :return:
        """
        if not self.exp_lresults:
            raise NameError("No results yet retrieved")
        if labels is None:
            labels = [l['experiment'] for l in self.lquery]

        msumtest = []
        msumval = []
        categories = []
        for i, exp_result in enumerate(self.exp_lresults):
            categories.extend([labels[i]]*len(self.selection))
            if type(summary) == int:
                msumtest.append(exp_result['test'][self.selection, summary])
                msumval.append(exp_result['validation'][self.selection, summary])
            elif summary == 'sum':
                msumtest.append(np.sum(exp_result['test'][self.selection], axis=1))
                msumval.append(np.sum(exp_result['validation'][self.selection], axis=1))
            else:
                msumtest.append(exp_result['test'][self.selection, 0])
                msumval.append(exp_result['validation'][self.selection, 0])

        f, p = f_oneway(*msumtest)

        print ('One-way ANOVA Test set')
        print ('=============')

        print ('F value:', f)
        print ('P value:', p, '\n')

        mc = MultiComparison(np.hstack(msumtest), categories)
        result = mc.tukeyhsd()
        print(result)
        print()

        f, p = f_oneway(*msumtest)

        print ('One-way ANOVA Validation set')
        print ('=============')

        print ('F value:', f)
        print ('P value:', p, '\n')

        mc = MultiComparison(np.hstack(msumval), categories)
        result = mc.tukeyhsd()
        print(result)

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
    from Wind.Private.DBConfig import mongolocal
    query1 = {"experiment": "MLP_s2s_2","status":"done"}
    query2 = {"experiment": "RNN_s2s","status":"done"}
    results = DBResults(conn=mongolocal)
    results.retrieve_results_compare(query1, query2)
    results.selected_size()
    results.sample(0.05)
    results.selected_size()
    results.plot_densplot_compare(plot='scatter', summary='sum')
    plt.show()

