"""
.. module:: UploadConfig

UploadConfig
*************

:Description: UploadConfig

    

:Authors: bejar
    

:Version: 

:Created on: 11/06/2018 13:45 

"""

from __future__ import print_function
import argparse
from time import time
import json
from Wind.Util import load_config_file
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
import glob
import numpy as np

__author__ = 'bejar'

def main():
    """Uploads configuration to the DB

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pend', action='store_true', default=False, help='change status to pending')
    args = parser.parse_args()

    if not args.pend:
        lfiles = glob.glob('res*.json')
        lfiles = sorted(lfiles)
    else:
        lfiles = glob.glob('*.json')
        lfiles = sorted(lfiles)

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    count = 0
    for file in lfiles:
        config = load_config_file(file, upload=True)

        if args.pend:
            print(config['_id'], config['data']['datanames'][0])
            col.update({'_id': config['_id']}, {'$set': {'status': 'pending'}})
        else:
            if 'results' in config:
                print(config['_id'], config['data']['datanames'][0], np.sum([v for _, v, _ in config['results']]))
            elif 'result' in config:
                print(config['_id'], config['data']['datanames'][0], np.sum([v for _, v, _ in config['result']]))

            col.update({'_id': config['_id']}, {'$set': {'status': 'done'}})

            if 'results' in config:
                col.update({'_id': config['_id']}, {'$set': {'result': config['results']}})
            elif 'result' in config:
                col.update({'_id': config['_id']}, {'$set': {'result': config['result']}})

            col.update({'_id': config['_id']}, {'$set': {'etime': config['etime']}})
            if 'btime' in config:
                col.update({'_id': config['_id']}, {'$set': {'btime': config['btime']}})
            else:
                col.update({'_id': config['_id']}, {'$set': {'btime': config['etime']}})
            col.update({'_id': config['_id']}, {'$set': {'host': 'minotauro'}})

        count += 1
    print(count, 'Processed')


if __name__ == '__main__':
    main()