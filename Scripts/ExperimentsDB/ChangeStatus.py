"""
.. module:: TransformData

TransformData
*************

:Description: Change Status

    Changes the status of one or several experiments

    --all selects all the experiment with the state passed
    --exp selects all the experiments of the type passed
    --id selects the experiment with the id passed
    --status new status for the selected experiments
    --testdb use the test database instead of the final database

:Authors: bejar
    

:Version: 

:Created on: 12/04/2018 10:21 

"""


import argparse
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient

from tqdm import tqdm

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ostatus', default=None,  help='Experiment status')
    parser.add_argument('--exp', default=None,  help='Experiment type')    
    parser.add_argument('--id', default=None, help='Experiment ID')
    parser.add_argument('--patt', default=None, help='Section pattern')
    parser.add_argument('--nstatus', help='Experiment status', default='pending')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocaltest

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    if args.ostatus is not None:
        configs = col.find({'status':args.ostatus, 'experiment':args.exp})
        for conf in tqdm(configs):
            col.update({'_id': conf['_id']}, {'$set': {'status': args.nstatus}})
    elif args.patt is not None:
        configs = col.find({'experiment':args.exp, 'site': {'$regex': f'^{args.patt}-.'}})
        for conf in tqdm(configs):
            col.update({'_id': conf['_id']}, {'$set': {'status': args.nstatus}})
    elif args.exp is not None:
        configs = col.find({'experiment':args.exp})
        for conf in tqdm(configs):
            col.update({'_id': conf['_id']}, {'$set': {'status': args.nstatus}})
    elif args.id is not None:
        col.update({'_id': args.id}, {'$set': {'status': args.nstatus}})
    else:
        raise NameError("Selection missing")


