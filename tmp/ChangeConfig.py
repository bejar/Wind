"""
.. module:: ChangeConfig

ChangeConfig
*************

:Description: ChangeConfig

    

:Authors: bejar
    

:Version: 

:Created on: 19/12/2018 11:39 

"""


import numpy as np
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient

__author__ = 'bejar'


if __name__ == '__main__':
    testdb = False
    update = True
    if testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp= 'CNN_s2s'

    configs = col.find({'experiment': exp, 'status': 'done'})

    count = 0
    for conf in configs:
        if update:
            col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
            col.update({'_id': conf['_id']}, {'$set': {'arch.mode': exp}})
            col.update({'_id': conf['_id']}, {'$set': {'training.epochs': 200}})
            col.update({'_id': conf['_id']}, {'$set': {'training.patience': 10}})
            col.update({'_id': conf['_id']}, {'$set': {'training.batch': 1024}})
            col.update({'_id': conf['_id']}, {'$set': {'training.iter': 1}})
            col.update({'_id': conf['_id']}, {'$set': {'data.datasize': 43834}})
            col.update({'_id': conf['_id']}, {'$set': {'data.testsize': 17534}})
            col.update({'_id': conf['_id']}, {'$set': {'data.ahead': [1,12]}})
            col.update({'_id': conf['_id']}, {'$set': {'data.lag': 18}})

        count += 1

    print(count)
