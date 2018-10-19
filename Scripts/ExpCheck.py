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
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
import argparse

__author__ = 'bejar'


if __name__ == '__main__':
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr', help='Experiment status',type=int,default=0)
    parser.add_argument('--to', help='Experiment status',type=int,default=253)
    parser.add_argument('--status', help='Experiment status',default='pending')
    args = parser.parse_args()

    total = 0
    for i in range(args.fr,args.to+1):
        configs = col.find({'experiment':'rnnseq2seq', 'status':args.status, 'site': {'$regex':'^%s-.'%str(i)}})
        count = 0
        lsites = []
        for conf in configs:
            count += 1
            lsites.append(conf['site'])
            #col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            # if conf['site'] in lsites:
            #     print(conf['_id'])
            #     col.remove({'_id':conf['_id']})
            # print(conf['site'])
#        for l in sorted(lsites):
#            print(l)
        total += count
        print (i, count)
    print(total, args.status)
