"""
.. module:: GenerateExpSites

GenerateExpSites
*************

:Description: GenerateExpSites

    

:Authors: bejar
    

:Version: 

:Created on: 07/06/2018 15:45 

"""

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient
import numpy as np


__author__ = 'bejar'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configrnnseq2seq', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--upper', type=int, help='Select upper best',default=100)
    parser.add_argument('--lower', type=int, help='Select lower worst',default=100)
    parser.add_argument('--exp',  help='Experiment type', default="eastwest9597")
    parser.add_argument('--mode', help='Experiment type', default='seq2seq')
    parser.add_argument('--suff', help='Datafile suffix', default='12')

    args = parser.parse_args()
    config = load_config_file(args.config)

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exps = col.find({'experiment': args.exp, 'arch.mode': args.mode})

    lexps = []
    for e in exps:
        lexps.append((np.sum(np.array(e['result'])[:,1]),  e['site']))


    lupper = [v for _,v in sorted(lexps, reverse=True)][:args.upper]
    llower = [v for _,v in sorted(lexps, reverse=False)][:args.lower]
    lexps = []
    lexps.extend(lupper)
    lexps.extend(llower)
    print(len(lexps))

    if args.test:
        for i, e in enumerate(lexps):
            print(i, e)

    else:
        ids = int(time())
        for i, site in enumerate(lexps):
            config['site'] =  site
            config['data']['datanames'] = ['%s-%s' % (site,args.suff)]
            config['status'] = 'pending'
            config['result'] = []
            config['_id'] = "%d%04d" % (ids, i)
            col.insert(config)
            print(config)


