"""
.. module:: MoveResults

MoveResults
*************

:Description: MoveResults

    

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 7:13 

"""
from Wind.Private.DBConfig import mongoconnection, mongolocal
from pymongo import MongoClient
import argparse

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='Experiment status', default='convos2s')
    parser.add_argument('--rexp', help='Experiment status', default=None)
    args = parser.parse_args()

    if args.rexp is None:
        args.rexp = args.exp

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    clientlocal = MongoClient(mongolocal.server)
    dblocal = clientlocal[mongolocal.db]
    collocal = dblocal[mongolocal.col]

    configs = col.find({'experiment': args.exp})

    for conf in configs:
        conf['experiment'] = args.rexp
        collocal.insert_one(conf)