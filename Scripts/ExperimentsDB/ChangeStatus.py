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
    parser.add_argument('--all', default='zz',  help='Experiment ID')    
    parser.add_argument('--id', help='Experiment ID')
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

    if args.all is not None:
        configs = col.find({'status':args.all})
        count = 0
        for conf in configs:
            if 'site' in conf:
                print(conf['_id'], conf['site'], conf['status'])
            else:
                print(conf['_id'], conf['status'])
            count += 1
            col.update({'_id': conf['_id']}, {'$set': {'status': args.status}})
        print(count)
    else:
        col.update({'_id': args.id}, {'$set': {'status': args.status}})


