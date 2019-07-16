"""
.. module:: Maps

Maps
*************

:Description: Maps

    Module for functions that generate maps

:Authors: bejar
    

:Version: 

:Created on: 21/06/2019 13:02 

"""

import plotly.offline as py
import plotly.graph_objs as go
from Wind.Private.DBConfig import mapbox_token


__author__ = 'bejar'

# Plotly colorscales
#
# [‘Blackbody’, ‘Bluered’, ‘Blues’, ‘Earth’, ‘Electric’, ‘Greens’, ‘Greys’, ‘Hot’, ‘Jet’,
# ‘Picnic’, ‘Portland’, ‘Rainbow’, ‘RdBu’, ‘Reds’, ‘Viridis’, ‘YlGnBu’, ‘YlOrRd’]

scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"], \
      [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"], \
      [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [1, "rgb(255, 0, 0)"]

sclbi = [0, "rgb(0,255,255)"], [0.49999, "rgb(0, 0, 255)"], [0.5, "rgb(0, 0, 0)"], [0.50001, "rgb(255, 0, 0)"], [1,
                                                                                                                 "rgb(255, 255, 0)"]
sclbi = [0, "rgb(0,0,255)"], [0.49999, "rgb(0, 255, 255)"], [0.5, "rgb(0, 0, 0)"], [0.50001, "rgb(255, 255, 0)"], [1,
                                                                                                                   "rgb(255, 0, 0)"]

sclbest = [0, "rgb(255,0,0)"], [0.1, "rgb(0,0,255)"], [0.2, "rgb(0,255,0)"], [0.3, "rgb(255,0,0)"], [0.4,
                                                                                                     "rgb(255,255,0)"], [
              0.5, "rgb(255,0,255)"], [0.4, "rgb(0,255,255)"]


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


def create_plot(df, title, notebook=False, tick=10, cmap=scl, figsize=(1200, 800), mksize=2):
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
            size=mksize,
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


def create_plot_best(df, title, labels, notebook=False, image=False, tick=10, cmap=None, figsize=(1200, 800)):
    """
    Creates an HTML file with the map
    """

    if scl is None:
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    else:
        colors = scl

    data = dict(
        mode='markers',
        type='scattergeo'
    )

    ldata = []
    for i in range(len(labels)):
        ndata = data.copy()
        ndata['lat'] = df[df['Val'] == i]['Lat']
        ndata['lon'] = df[df['Val'] == i]['Lon']
        ndata['marker'] = dict(size=1, color=i)
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