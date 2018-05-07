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

    configs = col.find({'status':'pending', 'arch.coef0':0,'arch.kernel':'poly','arch.degree':2})

    for conf in configs:
        col.update({'_id': conf['_id']}, {'$set': {'status': 'failed'}})


