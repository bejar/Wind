"""
.. module:: UploadConfig

UploadConfig
*************

:Description: UploadConfig

    

:Authors: bejar
    

:Version: 

:Created on: 11/06/2018 13:45 

"""


import argparse
from time import time
import json
from Wind.Util import load_config_file
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
import glob

__author__ = 'bejar'

if __name__ == '__main__':
    lfiles = glob.glob('res*.json')
    lfiles = sorted(lfiles)

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]


    for file in lfiles:
        config = load_config_file(file, upload=True)
        print(config['_id'])

        col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})
        col.update({'_id': config['_id']}, {'$set': {'result': config['results']}})
        col.update({'_id': config['_id']}, {'$set': {'etime': config['etime']}})
        if 'btime' in config:
            col.update({'_id': config['_id']}, {'$set': {'btime': config['btime']}})
        else:
            col.update({'_id': config['_id']}, {'$set': {'btime': config['etime']}})
        col.update({'_id': config['_id']}, {'$set': {'host': 'minotauro'}})

