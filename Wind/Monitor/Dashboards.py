"""
.. module:: Dashboards

Dashboards
*************

:Description: Dashboards

    

:Authors: bejar
    

:Version: 

:Created on: 21/03/2018 14:27 

"""
import dash
import dash_core_components as dcc
import dash_html_components as html

__author__ = 'bejar'


def data_options():
    return [{'label':i, 'value':i} for i in range(4)]

app = dash.Dash()
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
app.layout = html.Div(children=[
    html.H1(children='Wind Experiments'),
            html.Div([
                html.Div('Dataset', className='three columns'),
                html.Div(dcc.Dropdown(id='division-selector',
                                      options=data_options()),
                         className='nine columns'),
                html.Div('Dataset', className='three columns'),
                html.Div(dcc.Dropdown(id='division-selector',
                                      options=data_options()),
                         className='nine columns'),
            ]),
]
)

if __name__ == '__main__':
    app.run_server(debug=True, port=9010)
