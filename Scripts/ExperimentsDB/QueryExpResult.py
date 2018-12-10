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
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient



__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='10-5308',  help='Experiment site')    
    args = parser.parse_args()

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


