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
    parser.add_argument('--all', default=None, help='Experiment ID')    
    parser.add_argument('--id', help='Experiment ID')
    parser.add_argument('--status', help='Experiment status', default='pending')
    args = parser.parse_args()

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    if args.all is not None:
        configs = col.find({'status':args.all})
        count = 0
        for conf in configs:
            print(conf['_id'], conf['site'], conf['status'])
            count += 1
            col.update({'_id': conf['_id']}, {'$set': {'status': args.status}})
        print(count)
    else:
        col.update({'_id': args.id}, {'$set': {'status': args.status}})


