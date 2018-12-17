"""
.. module:: ExpStatus.py

ExpStatus.py
*************

:Description: ExpStatus.py

    

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 13:29 

"""
from __future__ import print_function
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from pymongo import MongoClient
import argparse

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'working'})
    jobs = [v for v in exp]
    print("Working = %d" % len(jobs))

    for i, j in enumerate(jobs):
        if 'btime' in j and 'host' in j:
            print('JOB %d = %s %s %s %s' % (i, j['_id'], j['btime'], j['site'], j['host']))
        else:
            print('JOB %d = ???' % i)

    exp = col.find({'status': 'pending'})
    print("Pending = %d" % len([v for v in exp]))

