"""
.. module:: UploadResults

UploadResults
*************

:Description: UploadResults

    Uploads all the json configuration files in the current directory

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 13:26 

"""
import argparse
import json

from pymongo import MongoClient

from Wind.Private.DBConfig import mongoconnection

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Experiments results')
    args = parser.parse_args()

    resfile = open('%s.json' % args.results, 'r')

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    for line in resfile:
        col.insert(json.loads(line))
