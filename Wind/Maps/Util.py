"""
.. module:: Util

Util
*************

:Description: Util

    

:Authors: bejar
    

:Version: 

:Created on: 15/06/2017 11:32 

"""

import folium
from geojson import LineString, FeatureCollection, Feature
import geojson
import numpy as np
from Wind.Config import wind_path

__author__ = 'bejar'


def MapThis(coords, ds, lfnames):
    coords = np.array(coords)
    mcoords = np.mean(coords, axis=0)
    print(mcoords)
    mymap = folium.Map(location=[mcoords[0], mcoords[1]], zoom_start=4, width=1200,
                       height=800)

    for c, f in zip(coords, lfnames):
        folium.Marker(c, popup=str(c) + '\n' + f).add_to(mymap)

    mymap.save(wind_path + 'Results/map%s.html'%ds)
