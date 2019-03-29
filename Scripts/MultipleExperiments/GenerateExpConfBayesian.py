"""
.. module:: GenerateExpConfBayesian

GenerateExpConf
*************

:Description: GenerateExpConfBayesian

    The configuration file has lists of values for all the parameters 
    instead of just the value

    Uploads to the DB a set of experiments based on previous estimations or 
    picks a set of random parameter combinations.

    Uses randomforest regressor to estimate the accuracy function over
    the parameters

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 12:29 

"""

import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongolocaltest
from copy import deepcopy
from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

__author__ = 'bejar'



def change_config(config, param, sites):
    """
    Substitutes default parameter values by generated parameters
    """
    lconf = []
    for s in sites:
        conf = deepcopy(config)
        for sec in param:
            for p in sec:
                conf[sec][p] = param[ses][p]
        conf['data']['site'] = [f'{s}-12']
        lconf.append(conf)
    return lconf
        
def generate_random_config(config):
    """
    generates a random config picking parameters from the configuration values
    """
    conf = {}
    for sec in config:
        conf[sec]={}
        for p in sec:
            conf[sec][p] =config[sec][p][np.random.choice(len(config[sec][p]))]
    return conf

def generate_configs(config):
    """
    Generates all possible individual configs from the fields with multiple values

    :param config:
    :return:
    """
    lconf = [{}]

    for f1 in config:
        for f2 in config[f1]:
            lnconf = []
            for v in config[f1][f2]:
                for c in lconf:
                    cp = deepcopy(c)
                    if f1 in cp:
                        cp[f1][f2] = v
                    else:
                        cp[f1] = {f2: v}
                    lnconf.append(cp)

            lconf = lnconf
    print('%d Configurations' % len(lconf))
    return lconf



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_MLP_s2s', help='Experiment configuration')
    parser.add_argument('--pconfig', default='pconfig_MLP_s2s', help='Paramters to explore configuratio')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--exp', required=True, help='Experiment Name')
    parser.add_argument('--npar', type=int, default=20, help='Number of parameter combinations')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    args = parser.parse_args()

    configB = load_config_file(args.config, upload=True)
    configP = load_config_file(args.config, upload=True)

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    # Find one site 
    sites = [col.find({'experiment': exp}, ['site'])]
    site = sites[0]
    # Get all unique configurations
    configurations = col.find({'experiment': exp, 'site':site})

    # Variables used from the experiments
    data = list(configP['data'].keys()) if 'data' in configP else []
    arch = list(configP['data'].keys()) if 'arch' in configP else []
    train = list(configP['data'].keys()) if 'train' in configP else []

    lvars = ['hour', 'site', 'test', 'val'] + data + arch + train
    ddata = {}
    for var in lvars:
        ddata[var] = []

    # Retrieve all results
    lexp =  col.find({'experiment': exp})
    for exp in lexp:
        # To maintain backwards compatibility
        if 'result' in exp:
            exdata = np.array(exp['result'])
        elif 'results' in exp:
            exdata = np.array(exp['results'])

        for i in range(exdata.shape[0]):
            lvals = [i+1]
            lvals.append(int(exp['data']['datanames'][0].split('-')[1]))
            lvals.append(exdata[i, 1])
            lvals.append(exdata[i, 2])

            for v in data:
                lvals.append(str(exp['data'][v]))

            for v in arch:
                lvals.append(str(exp['arch'][v]))

            for v in train:
                lvals.append(str(exp['training'][v]))

            for var, val in zip(lvars, lvals):
                ddata[var].append(val)
                
    exp_df = pd.DataFrame(ddata)
    exp_df = self.exp_df.groupby(by=['site']+arch + train + data,as_index=False).sum()
    exp_df.drop(columns=['hour', 'site'], inplace=True)
    exp_df = self.exp_df.groupby(by=+arch + train + data,as_index=False).agg({'test':['mean','count'], 'val':'mean', })

    
    if args.test:
        conf = generate_configs(configB)
        for c in conf:
            print(c)
    else:
        if args.testdb:
            mongoconnection = mongolocaltest
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        if mongoconnection.user is not None:
            db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for i, config in tqdm(enumerate(generate_configs(configB))):
            config['experiment'] = args.exp
            config['status'] = 'pending'
            site = config['data']['datanames'][0].split('-')
            config['site'] = '-'.join(site[:2])
            config['result'] = []
            config['_id'] = f"{ids}{i:05d}{int(site[1]):06d}"
            col.insert_one(config)
