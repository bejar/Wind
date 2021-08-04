"""
.. module:: Util

Util
*************

:Description: Util

    Utilities for spatial distribution of sites

:Authors: bejar
    

:Version: 

:Created on: 15/06/2017 11:32 

"""
try:
    import folium
except ImportError:
    _has_folium = False
else:
    _has_folium = True

import numpy as np
from sklearn.neighbors import KDTree

from Wind.Config import wind_path, MAX_SITES
from Wind.Config.Paths import wind_data_path

__author__ = 'bejar'

class SitesCoords:
    coords = None
    tree = None

    def __init__(self):
        self.coords = np.load(wind_data_path + '/Coords.npy')
        self.tree = KDTree(self.coords, leaf_size=1)


    def get_coords(self, site):
         return(self.coords[site])


    def get_direct_neighbors(self, site, radius):
        """
        return direct neighbors inside a radius
        :param site:
        :return:
        """
        neigh = self.tree.query_radius(self.coords[site, :].reshape(1, -1), r=radius, count_only=False, return_distance=False)[0]
        return neigh


def get_direct_neighbors(site, radius):
    """
    return direct neighbors inside a radius
    :param site:
    :return:
    """
    coords = np.load(wind_data_path + '/Coords.npy')
    tree = KDTree(coords, leaf_size=1)
    neigh = tree.query_radius(coords[site, :].reshape(1, -1), r=radius, count_only=False, return_distance=False)[0]
    return neigh


def get_closest_k_neighbors(site, radius, k):
    """
    return K closesr neighbors inside a radius
    :param site:
    :return:
    """
    coords = np.load(wind_data_path + '/Coords.npy')
    tree = KDTree(coords, leaf_size=1)
    isite = int(site.split('-')[1])
    agg = int(site.split('-')[2])

    lneighbor = tree.query_radius(coords[isite, :].reshape(1, -1), r=radius, sort_results=True, count_only=False, return_distance=True)[0][0]

    print([f"{v//500}-{v}-{agg}" for v in lneighbor[:k+1]])
    return [f"{v//500}-{v}-{agg}" for v in lneighbor[:k+1]]

def get_random_k_nonneighbors(site, radius, k):
    """
    Generates a list of random sites outside a radious of a given site
    """
    coords = np.load(wind_data_path + '/Coords.npy')
    tree = KDTree(coords, leaf_size=1)
    isite = int(site.split('-')[1])
    agg = int(site.split('-')[2])

    # retrieve a list of the 250 neighbors around a radius
    lneighbor = tree.query_radius(coords[isite, :].reshape(1, -1), r=radius, sort_results=True, count_only=False, return_distance=True)[0][0]
    lneighbor = set(lneighbor[:250])
    nnonneigh = 0
    lnonneighbor = []
    while nnonneigh <= k:
        rsite = np.random.randint(0, MAX_SITES)
        if rsite not in lneighbor:
            lnonneighbor.append(rsite)
            nnonneigh += 1

    return ["%d-%d-%d" % (v // 500, v, agg) for v in lnonneighbor[:k]]


def get_all_neighbors(site, radius):
    """
    Returns all site neighbors of a site inside a cluster

    site is the string of the datafile

    :param site:
    :return:
    """

    isite = int(site.split('-')[1])
    agg = int(site.split('-')[2])
    lneighbor = [isite]
    sneigh = set(lneighbor)

    new = True

    while new:
        cneigh = lneighbor.pop()
        nneigh = get_direct_neighbors(cneigh, radius)

        new = False
        for n in nneigh:
            if n not in sneigh:
                sneigh.add(n)
                lneighbor.append(n)
                new = True

    lneighbor = sorted(sneigh)
    lneighbor.remove(isite)
    tmp = [site]
    tmp.extend(["%d-%d-%d" % (v // 500, v, agg) for v in lneighbor])

    return tmp


def MapThis(coords, ds, lfnames):
    coords = np.array(coords)
    mcoords = np.mean(coords, axis=0)
    print(mcoords)
    mymap = folium.Map(location=[mcoords[0], mcoords[1]], zoom_start=4, width=1200,
                       height=800)

    for c, f in zip(coords, lfnames):
        folium.Marker(c, popup=str(c) + '\n' + f).add_to(mymap)

    mymap.save(wind_path + 'Results/map%s.html' % ds)


if __name__ == '__main__':
    # print(get_direct_neighbors(1203, 0.05))

    lneighbor = [4961]

    # print(get_all_neighbors("9-4961-12", 0.05))
    print(get_closest_k_neighbors("9-4961-12", 0.05,5))
