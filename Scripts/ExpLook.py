"""
.. module:: ExpCheck

ExpCheck
*************

:Description: ExpLook

    

:Authors: bejar
    

:Version: 

:Created on: 03/09/2018 8:50 

"""
import argparse
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', help='Experiment status')
    args = parser.parse_args()

    
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    total = 0
    configs = col.find({'status':args.status})
    count = 0
    lsites = []
    for conf in configs:
        count += 1
        lsites.append(conf['site'])
    print (args.status, count)
