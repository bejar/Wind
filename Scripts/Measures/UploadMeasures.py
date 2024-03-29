"""
.. module:: UploadMeasures

UploadMeasures
*************

:Description: UploadMeasures

    

:Authors: bejar
    

:Version: 

:Created on: 22/07/2019 10:18 

"""
import argparse
import glob
import os
import time

from pymongo import MongoClient
from tqdm import tqdm

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongoconnection, mongolocaltest

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()


    lfiles = glob.glob('measure*.json')
    lfiles = sorted(lfiles)
    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    count = 0
    for file in tqdm(lfiles):
        config = load_config_file(file, upload=True)
        col.insert_one(config)
        os.rename(file, f'done_{file}')

        count += 1
    print(f'{count} Processed {time.ctime()}')
