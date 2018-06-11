"""
.. module:: Util

Util
*************

:Description: Util

    

:Authors: bejar
    

:Version: 

:Created on: 07/02/2018 10:52 

"""

import json
from pymongo import MongoClient
from Wind.Private.DBConfig import mongoconnection
import numpy as np

__author__ = 'bejar'


def load_config_file(nfile, abspath=False, id=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    config = json.loads(s)
    if id:
        config['_id'] = '00000000'

    return config


def find_exp(query):
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    return col.find(query)


def count_exp(query):
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    print(col.count(query))


def sel_result(lexp, ncol):
    ldata = []
    for exp in lexp:
        data = np.array(exp['result'])
        ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, ncol]))
    ldata = sorted(ldata, key=lambda x: x[0])

    return np.array([v[0] for v in ldata]), np.array([v[1] for v in ldata])