"""
.. module:: SitesClustering

SitesClustering
*************

:Description: SitesClustering

    

:Authors: bejar
    

:Version: 

:Created on: 22/11/2018 11:46 

"""

from sklearn.cluster import KMeans
import numpy as np
from Wind.Config import wind_data_path
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS, SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
from Wind.Spatial.Util import SitesCoords

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.mixture import BayesianGaussianMixture

__author__ = 'bejar'

def create_plot(df, title):
    """
    """
    print(title)
    data = [dict(
        lat=df['Lat'],
        lon=df['Lon'],
        text=df['Val'].astype(str) + '-' + df['Site'].astype(str),
        marker=dict(
            color=df['Val'],
            colorscale=scl,
            reversescale=True,
            opacity=0.7,
            size=10,
            colorbar=dict(
                thickness=10,
                titleside="right",
                outlinecolor="rgba(68, 68, 68, 0)",
                ticks="outside",
                ticklen=3,
                showticksuffix="last",
                ticksuffix=" ",
                dtick=2
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
                type='conic conformal',
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
    # py.init_notebook_mode(connected=True)

    py.plot(fig, filename='./' + title + '.html')


def compute_clusterings(lsites, nc, mutual=False):
    """
    Computes the clustering of the sites days

    :param sites:
    :param nc:
    :return:
    """
    lclust = []

    for site in tqdm(lsites):
        nfile = wind_data_path + f'{site//500}-{site}-12.npy'

        data = np.load(nfile)
        wind = data[:,0]
        wind=wind.reshape(-1,24)

        kmeans = KMeans(n_clusters=nc)
        kmeans.fit(wind)
        if mutual:
            lclust.append((kmeans, wind))
        else:
            lclust.append(kmeans.cluster_centers_)
    return lclust

def compute_distance_matrix(lclust, mutual=False):
    """
    Computes a distance matrix with the sites

    :param lclust:
    :return:
    """
    nsites = len(lclust)
    mdist = np.zeros((nsites, nsites))

    for i in tqdm(range(nsites)):
        for j in range(i+1, nsites):
            if mutual:
                dist = clust_distance(lclust[i][0], lclust[i][1], lclust[j][0], lclust[j][1])
            else:
                dist = np.sum(np.min(euclidean_distances(lclust[i], lclust[j]), axis=0))
            mdist[i,j] = dist
            mdist[j,i] = mdist[i,j]

    return mdist

def plot_md_scaling(mdist, nd=3):
    """
    Plots the dimensionality rescaling of the distance matrix

    :param mdist:
    :param nd:
    :return:
    """
    #mds = MDS(n_components=3, dissimilarity='precomputed')

    mds = SpectralEmbedding(n_components=nd, affinity='precomputed', n_neighbors=3)
    pdata = mds.fit_transform(mdist)

    fig = plt.figure(figsize=(10,10))
    if nd == 2:
        ax = fig.add_subplot(111)
        ax.scatter(pdata[:, 0], pdata[:, 1],  cmap=plt.get_cmap("Blues"));
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pdata[:, 0], pdata[:, 1], pdata[:, 2], depthshade=False,  cmap=plt.get_cmap("Blues"))

    plt.show()

def md_scaling(mdist, nd=3):
    """
    performs dimensionality reduction of the distance matrix
    :param mdist:
    :param nd:
    :return:
    """
    nneig = int(np.sqrt(mdist.shape[0]))
    mds = SpectralEmbedding(n_components=nd, affinity='precomputed', n_neighbors=nneig)
    return mds.fit_transform(mdist)

def data_plot(lsites, labels):
    coords = np.array([scoords.get_coords(site) for site in lsites])
    return pd.DataFrame({'Lat': coords[:,0],
                   'Lon': coords[:,1],
                   'Val': labels,
                   'Site': np.array(lsites)})


def adjust_nc(data):
    """
    adjusts the number of clusters using silhouette

    :param data:
    :return:
    """
    sil = []
    for nc in range(2,int(np.sqrt(data.shape[0]))):
        km = KMeans(n_clusters=nc, n_init=30)
        labels = km.fit_predict(data)
        sil.append(silhouette_score(data, labels))

    return np.argmax(sil)+1


def clust_distance(c1, d1, c2, d2):
    """
    Clustering distance as mutual information
    """

    l11 = c1.predict(d1)
    l21 = c2.predict(d1)
    l22 = c2.predict(d2)
    l12 = c1.predict(d2)
    return mutual_info_score(l11,l21) + mutual_info_score(l22, l12)


scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"], \
      [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"], \
      [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [1, "rgb(255, 0, 0)"]


if __name__ == '__main__':

    scoords = SitesCoords()
    sites_i = 39000
    sites_f = 12100
    nc = 50
    mutual = True
    lsites = scoords.get_direct_neighbors(sites_i, 0.65)
    # lsites = range(sites_i, sites_f)
    lclust = compute_clusterings(lsites, nc, mutual=mutual)
    mdist = compute_distance_matrix(lclust, mutual=mutual)
    #plot_md_scaling(mdist)
    tdata = md_scaling(mdist)

    #cs = adjust_nc(tdata)
    #kmeans = KMeans(n_clusters=cs)
    #labels = kmeans.fit_predict(tdata)

    gmm = BayesianGaussianMixture(n_components=10, covariance_type='full')
    labels = gmm.fit_predict(tdata)
    create_plot(data_plot(lsites, labels), str(sites_i))
