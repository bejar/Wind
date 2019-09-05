"""
.. module:: ExpCheck

ExpCheck
*************

:Description: ExpCheck

:Authors: bejar
    

:Version: 

:Created on: 03/09/2018 8:50 

"""
from __future__ import print_function
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from pymongo import MongoClient
import argparse

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiment name', default='convos2s')
    parser.add_argument('--status', help='Experiment status', default='pending')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    total = 0
    configs = col.find({'experiment': args.exp, 'status': args.status})
    for conf in configs:
        #print(conf['result'][0][0])
        if conf['result'][0][1] < 0.11:
            print(conf['site'])
