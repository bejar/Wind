"""
.. module:: MoveResults

MoveResults
*************

:Description: MoveResults

    Moves experiments from one database to another

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
    parser.add_argument('--exp', help='Experiment type', default='convos2s')
    parser.add_argument('--rexp', help='Experiment status', default=None)
    parser.add_argument('--ptt', action='store_true', default=False, help='Move production to test')
    args = parser.parse_args()

    if args.rexp is None:
        args.rexp = args.exp

    if args.ptt:
        mprod = mongolocal
        mtest = mongolocaltest
    else:
        mprod = mongolocaltest
        mtest = mongolocal


    client = MongoClient(mprod.server)


    db = client[mprod.db]
    if mprod.user is not None:
        db.authenticate(mprod.user, password=mprod.passwd)
    col = db[mprod.col]


    clientlocal = MongoClient(mtest.server)

    dblocal = clientlocal[mtest.db]
    collocal = dblocal[mtest.col]

    configs = col.find({'experiment': args.exp})


    for conf in tqdm(configs):
        if args.rexp:
            conf['experiment'] = args.rexp

        confex = collocal.find_one({'_id': conf['_id']})
        if confex is not None:
            collocal.delete_one({'_id': conf['_id']})

        collocal.insert_one(conf)

