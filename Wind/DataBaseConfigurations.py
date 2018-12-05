"""
.. module:: Training

Training
*************

:Description: Training

    

:Authors: bejar
    

:Version: 

:Created on: 06/04/2018 14:32 

"""
import requests
from time import strftime
import socket
import json
import numpy as np
from Wind.Config.Paths import wind_res_path

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo = False
else:
    _has_mongo = True

__author__ = 'bejar'


def getconfig(proxy=False, mode=None, secpat=None):
    """
    Gets a config from the database
    :return:
    """
    if not proxy:
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]
        query = {'status': 'pending'}
        if secpat is not None:
            query['site'] = {'$regex': '%s-.' % secpat}
        if mode is not None:
            query['experiment'] = mode

            # config = col.find_one(query)

        lconfig = [c for c in col.find(query, limit=100)]
        config = None
        if len(lconfig) > 0:
            ch = np.random.randint(0, len(lconfig))
            for i, conf in enumerate(lconfig):
                if i == ch:
                    config = conf

        if config is not None:
            col.update({'_id': config['_id']}, {'$set': {'status': 'working'}})
            col.update({'_id': config['_id']}, {'$set': {'btime': strftime('%Y-%m-%d %H:%M:%S')}})
            col.update({'_id': config['_id']}, {'$set': {'host': socket.gethostname().split('.')[0]}})
        return config
    else:
        return requests.get('http://polaris.cs.upc.edu:9073/Proxy', params={'mode': mode}).json()


def failconfig(config, proxy=False):
    """
    Changes the status of the configuration to pending if the experiment fails
    :param config:
    :param lresults:
    :param proxy:
    :return:
    """
    if not proxy:
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]
        col.update({'_id': config['_id']}, {'$set': {'status': 'pending'}})


def saveconfig(config, lresults, proxy=False, mino=False):
    """
    Saves a config in the database
    :param proxy:
    :return:
    """

    if not proxy:
        if not mino:
            client = MongoClient(mongoconnection.server)
            db = client[mongoconnection.db]
            db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
            col = db[mongoconnection.col]

            if lresults[0][1] > 0.0:
                col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})
                col.update({'_id': config['_id']}, {'$set': {'result': lresults}})
                col.update({'_id': config['_id']}, {'$set': {'etime': strftime('%Y-%m-%d %H:%M:%S')}})
            else:
                col.update({'_id': config['_id']}, {'$set': {'status': 'pending'}})
        else:
            config['status'] = 'done'
            config['result'] = lresults
            config['etime'] = strftime('%Y-%m-%d %H:%M:%S')
            sconf = json.dumps(config)
            fconf = open(wind_res_path + '/res' + config['_id'] + '.json', 'w')
            fconf.write(sconf + '\n')
            fconf.close()


    else:
        config['result'] = lresults
        requests.post('http://polaris.cs.upc.edu:9073/Proxy', params={'res': json.dumps(config)})


def updateprocess(config, ahead, proxy=False):
    """
    updates the info of the training process for each iteration

    :param config:
    :return:
    """
    if not proxy and _has_mongo and config['_id'] != '00000000':
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]
        col.update({'_id': config['_id']}, {'$set': {'ahead': ahead}})
        col.update({'_id': config['_id']}, {'$set': {'etime': strftime('%Y-%m-%d %H:%M:%S')}})


if __name__ == '__main__':
    print(getconfig(rsec='20.'))
