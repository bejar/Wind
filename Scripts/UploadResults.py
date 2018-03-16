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
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Experiments results')
    args = parser.parse_args()

    resfile = open('%s.json' % args.results, 'r')

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    for line in resfile:
        col.insert(json.loads(line))

