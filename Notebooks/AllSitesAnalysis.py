import matplotlib
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from pylab import *
from matplotlib import rc
import matplotlib.ticker as ticker
import seaborn as sn
import folium

from Wind.Util import find_exp, count_exp, sel_result
from Wind.Config.Paths import wind_data_path

import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd

def create_plot(df, title):
    """
    """
    print(title)
    data = [ dict(
        lat = df['Lat'],
        lon = df['Lon'],
        text = df['Val'].astype(str)+'-'+df['Site'].astype(str),
        marker = dict(
            color = df['Val'],
            colorscale = scl,
            reversescale = True,
            opacity = 0.7,
            size = 2,
            colorbar = dict(
                thickness = 10,
                titleside = "right",
                outlinecolor = "rgba(68, 68, 68, 0)",
                ticks = "outside",
                ticklen = 3,
                showticksuffix = "last",
                ticksuffix = " ",
                dtick = 2
            ),
        ),
        type = 'scattergeo'
    ) ]

    layout = dict(
        geo = dict(
            scope = 'north america',
            showland = True,
            landcolor = "rgb(212, 212, 212)",
            subunitcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(255, 255, 255)",
            showlakes = True,
            lakecolor = "rgb(255, 255, 255)",
            showsubunits = True,
            showcountries = True,
            resolution = 50,
            projection = dict(
                type = 'conic conformal',
                rotation = dict(
                    lon = -100
                )
            ),
            lonaxis = dict(
                showgrid = True,
                gridwidth = 0.5,
                range= [ -140.0, -55.0 ],
                dtick = 5
            ),
            lataxis = dict (
                showgrid = True,
                gridwidth = 0.5,
                range= [ 20.0, 60.0 ],
                dtick = 5
            )
        ),
        width=1200,
        height=800,
        title = title,
    )
    fig = { 'data':data, 'layout':layout }
    #py.init_notebook_mode(connected=True)

    py.plot(fig, filename='./' + title + '.html')


cpal = plt.get_cmap('Reds')

coords = np.load(wind_data_path +'/coords.npy')

scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\
      [0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\
      [0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]

if __name__ == '__main__':

    #------------------------------------------
    # Persistence
    query1= {'status':'done', "experiment": "Persistence", "site": {"$regex":"."}}

    count_exp(query1)
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,1)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum

    print(sites1.shape)
    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]),
                       'Lon': np.append(coords[sites1,1],[0,0]),
                       'Val':np.append(np.sum(coord1, axis=1),[12,-7]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'persistence1')
    
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,2)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum

    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]),
                       'Lon': np.append(coords[sites1,1],[0,0]),
                       'Val':np.append(np.sum(coord1, axis=1),[12,-7]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'persistence2')

    # -------------------------------------------
    # MLP multiple regression
    query1= {'status':'done', "experiment": "mlpregs2s", "site": {"$regex":"."}}

    count_exp(query1)
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,1)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum


    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]), 'Lon': np.append(coords[sites1,1],[0,0]), 'Val':np.append(np.sum(coord1, axis=1),[10,1]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'MLPRegS2S1')

    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,2)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum


    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]), 'Lon': np.append(coords[sites1,1],[0,0]), 'Val':np.append(np.sum(coord1, axis=1),[10,1]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'MLPRegS2S2')

    # --------------------------------------
    # RNN seq2seq

    query1= {'status':'done', "experiment": "rnnseq2seq", "site": {"$regex":"."}}

    count_exp(query1)
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,1)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum

    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]),
                       'Lon': np.append(coords[sites1,1],[0,0]),
                       'Val':np.append(np.sum(coord1, axis=1),[10,1]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'RNNS2S1')

    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1,2)

    valsum = np.sum(coord1,axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum

    df = pd.DataFrame({'Lat': np.append(coords[sites1,0],[0,0]), 'Lon': np.append(coords[sites1,1],[0,0]), 'Val':np.append(np.sum(coord1, axis=1),[10,1]),
                       'Site':np.append(sites1,[0,0])})

    create_plot(df, 'RNNS2S2')

