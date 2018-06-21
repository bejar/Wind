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
import numpy as np
from time import strftime

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo= False
else:
    _has_mongo = True

__author__ = 'bejar'


def load_config_file(nfile, abspath=False, id=False, upload=False):
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

    if not upload:
        config['btime'] = strftime('%Y-%m-%d %H:%M:%S')

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


def sel_upper_lower(exp, mode, column, upper=100, lower=100):
    """

    :param up:
    :param low:
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exps = col.find({'experiment': exp, 'arch.mode': mode})

    lexps = []
    for e in exps:
        lexps.append((np.sum(np.array(e['result'])[:, column]), e['site'], np.array(e['result'])[:, column]))

    lupper = [(v.split('-')[1], s) for _, v, s in sorted(lexps, reverse=True)][:upper]
    llower = [(v.split('-')[1], s)  for _, v, s in sorted(lexps, reverse=False)][:lower]
    lexps = []
    lexps.extend(lupper)
    lexps.extend(llower)

    return np.array([v[0] for v in lexps]), np.array([v[1] for v in lexps])

if __name__ == '__main__':
    sites1, coord1 = sel_upper_lower('eastwest9597', 'seq2seq',1)
    print(sites1)