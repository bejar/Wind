"""
.. module:: GenerateExpSample

GenerateExpSites
*************

:Description: GenerateRangeSample

    Generates and uploads to the DB configurations using --config configuration

    If --sample exists, it generates the configuration using the existing sample,
    if not it generates a sample of --nsamp sites and generates the configurations



:Authors: bejar
    

:Version: 

:Created on: 07/06/2018 15:45 

"""

import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient
from tqdm import tqdm


# nsites  0 - 126691
__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=True, help='Experiment configuration')
    parser.add_argument('--exp', default=None, required=True, help='Experiment name')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--sample', type=str, required=True, help='Name of the sample')
    parser.add_argument('--nsamp', type=int, default=2000, help='Size of the sample')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    args = parser.parse_args()

    config = load_config_file(args.config)

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    # Check if experiment already exists
    exp = col.find_one({'experiment': args.exp})
    if exp is not None:
        raise NameError("Experiment already exists")

    # Check if sample already exists
    sample = col.find_one({'sample': 'init', 'sname': args.sample})

    if sample is not None:
        ssites =  sample['sites']
    else:
        lsites = np.random.choice(range(126692), args.nsamp, replace=False)
        lsites = [f'{site // 500}-{site}' for site in lsites]
        sampleconf = {'sample': 'init', 'sname': args.sample, 'sites': lbatches}
        col.insert_one(sampleconf)
        ssites = lsites

    ids = int(time())
    for i, site in tqdm(enumerate(range(args.isite, args.fsite + 1))):
        config['data']['datanames'] = [site]
        site = config['data']['datanames'][0].split('-')
        config['site'] = '-'.join(site[:2])
        config['status'] = 'pending'
        config['experiment'] = args.exp
        config['result'] = []
        config['_id'] = f"{ids}{i:05d}{int(site[1]):06d}"
        col.insert_one(config)

