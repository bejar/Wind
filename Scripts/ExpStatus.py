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
from Wind.Private.DBConfig import mongoconnection
from copy import deepcopy
from pymongo import MongoClient

__author__ = 'bejar'

if __name__ == '__main__':
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'done'})
    print("Done = %d" % len([v for v in exp]))
    exp = col.find({'status': 'working'})
    print("Working = %d" % len([v for v in exp]))
    exp = col.find({'status': 'pending'})
    print("Pending = %d" % len([v for v in exp]))


