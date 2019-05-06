"""
.. module:: MoveResults

MoveResults
*************

:Description: MoveResults

    Duplicates all the jobs of an experiment with empty results and pending status

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 7:13 

"""
from Wind.Private.DBConfig import mongoconnection, mongolocal, mongolocaltest
from pymongo import MongoClient
import argparse
from tqdm import tqdm

__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help='Experiment type')
    parser.add_argument('--dexp', required=True, help='Duplicated Experiment')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]


    configs = col.find({'experiment': args.exp})

    ids = int(time()*10000)
    for n, conf in tqdm(enumerate(configs)):
        conf['timestamp'] = ids
        conf['result'] = []
        conf['experiment'] = args.dexp
        conf['status'] = 'pending'
        site = conf['data']['datanames'][0].split('-')
        conf['_id'] = f"{ids}{n:05d}{int(site[1]):06d}"

        collocal.insert_one(conf)

