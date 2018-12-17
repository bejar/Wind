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
    parser.add_argument('--fr', help='Experiment status', type=int, default=0)
    parser.add_argument('--to', help='Experiment status', type=int, default=253)
    parser.add_argument('--exp', help='Experiment status', default='convos2s')
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
    for i in range(args.fr, args.to + 1):
        configs = col.find({'experiment': args.exp, 'status': args.status, 'site': {'$regex': '^%s-.' % str(i)}})
        count = 0
        lsites = []
        for conf in configs:
            count += 1
            lsites.append(conf['site'])
            # col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            # if conf['site'] in lsites:
            #     print(conf['_id'])
            #     col.remove({'_id':conf['_id']})
            # print(conf['site'])
        #        for l in sorted(lsites):
        #            print(l)
        total += count
        print(i, count)
    print(total, args.status)
