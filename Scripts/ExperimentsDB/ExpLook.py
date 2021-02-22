"""
.. module:: ExpLook

ExpCheck
*************

:Description: ExpLook

    Counts the experiments that have a specific status

    --status status of the experiment
    --testdb use the test database instead of the final database


:Authors: bejar
    

:Version: 

:Created on: 03/09/2018 8:50 

"""
import argparse
from Wind.Private.DBConfig import mongolocaltest, mongoconnection, mongolocalmeasures
from pymongo import MongoClient

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=None, help='Type of experiment')
    parser.add_argument('--status', help='Experiment status')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    if args.testdb:
        mongoconnection = mongolocalmeasures
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    if args.exp is not None:
        total = 0
        configs = col.find({'status': args.status, 'experiment':args.exp})
        count = 0
        lsites = []
        for conf in configs:
            count += 1
            if 'site' in conf:
                lsites.append(conf['site'])
            else:
                lsites.append(conf['_id'])
        print(args.status, count)
    else:
        count = {}
        configs = col.find({'status': args.status})
        for conf in configs:
            if conf['experiment'] in count:
                count[conf['experiment']] += 1
            else:
                count[conf['experiment']] = 1

        if len(count) != 0:
            for exp in count:
                print(args.status, exp, count[exp])
        else:
            print('No more experiments')

