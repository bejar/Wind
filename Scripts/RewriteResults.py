"""
.. module:: UploadResults

UploadResults
*************

:Description: UploadResults

    

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 13:26 

"""

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Private.DBConfig import mongoconnection
from copy import deepcopy
from pymongo import MongoClient
import json

__author__ = 'bejar'

if __name__ == '__main__':
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    configs = col.find({'experiment':'eastwest9597', 'arch.mode':'regdir'})

    count = 0
    for conf in configs:
        col.update({'_id': conf['_id']}, {'$set': {'status': 'pending'}})
        col.update({'_id': conf['_id']}, {'$set': {'arch.full': [1]}})
        col.update({'_id': conf['_id']}, {'$set': {'training.epochs': 150}})
        count += 1

    print(count)



