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
from time import strftime

import numpy as np
from numpy import log, polyfit, sqrt, std, subtract

from Wind.Config.Paths import wind_jobs_path, wind_local_jobs_path

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo = False
else:
    _has_mongo = True

__author__ = 'bejar'


def load_config_file(nfile, abspath=False, id=False, upload=False, mino=False, local=False):
    """
    Reads a configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    if not mino:
        pre = '' if abspath else './'
    elif local:
        pre = wind_jobs_path
    else:
        pre = wind_local_jobs_path
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
    """
    Returns all the experiments in the DB that match the query

    :param query:
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    return col.find(query)


def count_exp(query):
    """
    Counts how many experiments in the DB match the query

    :param query:
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    print(col.count(query))


def sel_result(lexp, ncol):
    """
    Selects from a list of configurations with results the result in the column 0 (test) or 1 (validation)

    :param lexp:
    :param ncol:
    :return:
    """
    ldata = []
    for exp in lexp:

        # To maintain backwards compatibility
        if 'result' in exp:
            data = np.array(exp['result'])
        elif 'results' in exp:
            data = np.array(exp['results'])

        ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, ncol]))
    ldata = sorted(ldata, key=lambda x: x[0])

    return np.array([v[0] for v in ldata]), np.array([v[1] for v in ldata])


def sel_upper_lower(exp, mode, column, upper=100, lower=100):
    """
    Selects a number of experiment with the best and worst accuracy

    column selects test (0) and validation (1)
    :param up:
    :param low:
    :return:
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exps = col.find({'experiment': exp, 'arch.mode': mode})

    lexps = []
    for e in exps:
        if 'result' in e:
            lexps.append((np.sum(np.array(e['result'])[:, column]), e['site'], np.array(e['result'])[:, column]))
        elif 'results' in e:
            lexps.append((np.sum(np.array(e['results'])[:, column]), e['site'], np.array(e['results'])[:, column]))

    lupper = [(v.split('-')[1], s) for _, v, s in sorted(lexps, reverse=True)][:upper]
    llower = [(v.split('-')[1], s) for _, v, s in sorted(lexps, reverse=False)][:lower]
    lexps = []
    lexps.extend(lupper)
    lexps.extend(llower)

    return np.array([v[0] for v in lexps]), np.array([v[1] for v in lexps])


def SampEn(U, m, r):
    """
    Sample entropy, taken from wikipedia, probably can be improved using numpy functions
    :param U: The time series
    :param m: The embedding dimension
    :param r: the radius distance
    """

    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))


def hurst(ts, maxlag=200):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, maxlag)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

if __name__ == '__main__':

    count_exp({'experiment':'Persistence'})
