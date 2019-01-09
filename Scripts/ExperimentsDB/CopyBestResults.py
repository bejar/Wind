"""
.. module:: CopyBestResults

MoveResults
*************

:Description: CopyBestResults

    Compies the best results for an experiment from the testdb to the resultsdb

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 7:13 

"""
from Wind.Private.DBConfig import mongoconnection, mongolocal, mongolocaltest
from pymongo import MongoClient
import argparse
import numpy as np

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiment status', default='convos2s')
    args = parser.parse_args()


    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    clientt = MongoClient(mongolocaltest.server)
    dbt = client[mongoconnection.db]
    if mongolocaltest.user is not None:
        db.authenticate(mongolocaltest.user, password=mongolocaltest.passwd)
    colt = db[mongolocaltest.col]


    configs = colt.find({'experiment': args.exp})

    count = 0
    cbest = 0
    for conft in configs:
        count +=1
        conf = col.find_one({'experiment': args.exp, 'site':conft['site']})

        res = np.sum([v[1] for v in conf['result']])
        rest = np.sum([v[1] for v in conft['result']])
        if rest>res:
            print(f"{conft['site']} {rest-res}")
            cbest +=1

    print(f'{cbest} of {count}')