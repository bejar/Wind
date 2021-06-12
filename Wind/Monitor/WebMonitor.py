"""
.. module:: WebMonitor

WebMonitor
******

:Description: WebMonitor

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  18/03/2018
"""

import json
import socket
from time import strftime

import StringIO
import matplotlib
from flask import Flask, render_template, request
from flask.logging import default_handler
from pymongo import MongoClient

from Wind.Private.DBConfig import mongoconnection

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

import base64
import numpy as np

__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 9073

app = Flask(__name__)


def getconfig(mode=None):
    """
    Gets a config from the database
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    query = {'status': 'pending'}

    lconfig = [c for c in col.find(query, limit=100)]

    config = None
    if len(lconfig) > 0:
        ch = np.random.randint(0, len(lconfig))
        for i, conf in enumerate(lconfig):
            if i == ch:
                config = conf

    if config is not None:
        print("Served job %s#%s" % (config['_id'], config['data']['datanames'][0]))
        col.update({'_id': config['_id']}, {'$set': {'status': 'working'}})
        col.update({'_id': config['_id']}, {'$set': {'btime': strftime('%Y-%m-%d %H:%M:%S')}})
        col.update({'_id': config['_id']}, {'$set': {'host': 'proxy'}})

    return config


def saveconfig(config):
    """
    Saves a config in the database
    :param proxy:
    :return:
    """
    print("received job %s" % config['_id'])
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    if config['results'][0][1] > 0.5:
        col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})
        col.update({'_id': config['_id']}, {'$set': {'result': config['results']}})
        col.update({'_id': config['_id']}, {'$set': {'etime': strftime('%Y-%m-%d %H:%M:%S')}})
    else:
        col.update({'_id': config['_id']}, {'$set': {'status': 'pending'}})


@app.route('/Monitor')
def info():
    """
    Job status
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'done'})
    done = len([v for v in exp])

    exp = col.find({'status': 'working'})
    work = {}
    for v in exp:
        work[v['_id']] = {'mode': v['arch']['mode'],
                          'btime': v['btime'],
                          'host': v['host'] if 'host' in v else 'host',
                          'etime': v['etime'] if 'etime' in v else v['btime'],
                          'ahead': v['ahead'] if 'ahead' in v else 0}

    exp = col.find({'status': 'pending'})
    pend = len([v for v in exp])

    return render_template('Monitor.html', done=done, pend=pend, work=work)


@app.route('/Proxy', methods=['GET', 'POST'])
def proxy():
    """
    Proxy to configurations
    :return:
    """
    if request.method == 'GET':
        config = getconfig()
        return json.dumps(config)
    else:
        param = request.args['res']
        res = json.loads(param)
        saveconfig(res)
        return "OK"


@app.route('/Done', methods=['GET', 'POST'])
def done():
    """
    Done jobs
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'done', 'data.dataset': int(request.form['problem'])})
    done = {}
    for v in exp:
        done[v['_id']] = {'data': v['data']['datanames'][0],
                          'dataset': v['data']['dataset'],
                          'vars': len(v['data']['vars']),
                          'lag': v['data']['lag'],
                          'rnn': v['arch']['rnn'],
                          'bi': v['arch']['bimerge'] if v['arch']['bidirectional'] else 'no',
                          'nly': v['arch']['nlayers'],
                          'nns': v['arch']['neurons'],
                          'drop': v['arch']['drop'],
                          'act': v['arch']['activation'],
                          'opt': v['training']['optimizer'],
                          }

    return render_template('Done.html', done=done, prob=request.form['problem'])


@app.route('/Elist', methods=['GET', 'POST'])
def iface():
    """
    Interfaz con el cliente a traves de una pagina de web
    """
    probtypes = [0, 1, 2, 3]
    return render_template('ExpList.html', types=probtypes)


@app.route('/Experiment/<exp>')
def experiment(exp):
    """
    Individual Experiment

    :param exp:
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    expres = col.find_one({'_id': exp})

    data = np.array(expres['result'])

    img = StringIO.StringIO()
    fig = plt.figure(figsize=(16, 10), dpi=100)

    axes = fig.add_subplot(1, 1, 1)

    axes.plot(data[:, 0], data[:, 1], color='r')
    axes.plot(data[:, 0], data[:, 2], color='r', linestyle='--')
    axes.plot(data[:, 0], data[:, 3], color='g')
    axes.plot(data[:, 0], data[:, 4], color='g', linestyle='--')

    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue())
    plt.close()

    return render_template('Experiment.html', exp=expres, plot_url=plot_url)


if __name__ == '__main__':
    # The Flask Server is started
    default_handler.setLevel(logging.WARNING)
    app.run(host='0.0.0.0', port=port, debug=False)
