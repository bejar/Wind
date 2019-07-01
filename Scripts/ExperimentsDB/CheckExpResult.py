"""
.. module:: QueryExpResult

TransformData
*************

:Description: QueryExpResult

    Queries for the results of one site

    --site site to query
    --testdb use the test database instead of the final database

:Authors: bejar
    

:Version: 

:Created on: 12/04/2018 10:21 

"""


import argparse
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdb', action='store_true', default=True, help='Use test database')
    parser.add_argument('--exp', help='Experiment name', default='MLP_s2s_best')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    configs = col.find({'experiment': args.exp})


    for conf in configs:
        for v in conf['result']:
            if v[1] <0:
                print(conf['site'])
                print('****************************************')
                break


