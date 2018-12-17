"""
.. module:: TransformData

TransformData
*************

:Description: TransformData

    

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
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--site', default='10-5308',  help='Experiment site')    
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    configs = col.find({'site':args.site})
    for conf in configs:
        print('TYPE=', conf['arch']['mode'])
        for v in conf['result']:
            print(v)
        print('****************************************')


