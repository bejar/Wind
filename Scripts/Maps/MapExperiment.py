"""
.. module:: MapExperiment

GenerateExpConf
*************

:Description: MapExperiment

    Generates two HTML maps for the test and validation results

:Authors: bejar


:Version:

:Created on: 16/03/2018 12:29

"""

from pylab import *

from Wind.Misc import find_exp, count_exp, sel_result
from Wind.Config.Paths import wind_data_path

import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import argparse


def create_plot(df, title):
    """
    Creates an HTML file with the map
    """
    print(title)
    data = [dict(
        lat=df['Lat'],
        lon=df['Lon'],
        text=df['Val'].astype(str) + '-[' + df['Site'].astype(str) + ']',
        marker=dict(
            color=df['Val'],
            colorscale=scl,
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

    py.plot(fig, filename='./' + title + '.html')


cpal = plt.get_cmap('Reds')

scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"], \
      [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"], \
      [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [1, "rgb(255, 0, 0)"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='Persistence', help='experiment')
    args = parser.parse_args()

    coords = np.load(wind_data_path + '/Coords.npy')

    query1 = {'status': 'done', "experiment": args.exp, "site": {"$regex": "."}}

    print(f"Results ={count_exp(query1)}")

    # Map TEST data
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1, 1)

    valsum = np.sum(coord1, axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum
    df = pd.DataFrame({'Lon': np.append(coords[sites1, 0], [0, 0]),
                       'Lat': np.append(coords[sites1, 1], [0, 0]),
                       'Val': np.append(np.sum(coord1, axis=1), [10, 1]),
                       'Site': np.append(sites1, [0, 0])})

    create_plot(df, args.exp + '1')

    # Map VALIDATION data
    res1 = find_exp(query1)
    sites1, coord1 = sel_result(res1, 2)

    valsum = np.sum(coord1, axis=1)
    minsum = np.min(valsum)
    rangesum = np.max(valsum) - minsum

    df = pd.DataFrame({'Lon': np.append(coords[sites1, 0], [0, 0]),
                       'Lat': np.append(coords[sites1, 1], [0, 0]),
                       'Val': np.append(np.sum(coord1, axis=1), [10, 1]),
                       'Site': np.append(sites1, [0, 0])})

    create_plot(df, args.exp + '2')
