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

import socket

from flask import Flask, render_template, request
from pymongo import MongoClient
from Wind.Private.DBConfig import mongoconnection
from time import time, strftime
import json

__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 9000

app = Flask(__name__)


def getconfig():
    """
    Gets a config from the database
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    config = col.find_one({'status': 'pending'})
    if config is not None:
        print("Served job %s" % config['_id'])
        col.update({'_id': config['_id']}, {'$set': {'status': 'working'}})
        col.update({'_id': config['_id']}, {'$set': {'btime': strftime('%Y-%m-%d %H:%M:%S')}})

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
    col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})
    col.update({'_id': config['_id']}, {'$set': {'result': config['results']}})
    col.update({'_id': config['_id']}, {'$set': {'etime': strftime('%Y-%m-%d %H:%M:%S')}})

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
        work[v['_id']] = v['btime']

    exp = col.find({'status': 'pending'})
    pend= len([v for v in exp])

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

    print(request.form['problem'])
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'done', 'data':{'dataset':int(request.form['problem'])}})
    done = {}
    for v in exp:
        done[v['_id']] = {'data': v['data']['datanames'][0],
                          'dataset':v['data']['dataset'],
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
        print(v['_id'])

    return render_template('Done.html', done=done, prob=request.form['problem'])

@app.route('/Elist', methods=['GET', 'POST'])
def iface():
    """
    Interfaz con el cliente a traves de una pagina de web
    """
    probtypes = [0,1,2,3]
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

    return render_template('Experiment.html', exp=expres)


if __name__ == '__main__':
    # The Flask Server is started
    app.run(host='0.0.0.0', port=port, debug=False)
