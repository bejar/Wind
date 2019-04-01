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
            for p in param[sec]:
                conf[sec][p] = param[sec][p]
        conf['data']['site'] = [f'{s}-12']
        conf['site'] = s
        lconf.append(conf)
    return lconf

def hash_config(conf):
    """
    Creates a canonical string representation for a configuration
    by traversing alphabetically the dictionarty and appending a representation string
    :param conf:
    :return:
    """
    s = ""
    for sec in sorted(conf.keys()):
        for p in sorted(conf[sec].keys()):
            s+=f"{p}={conf[sec][p]}/"
        s+='#'
    return s


def generate_random_config(config):
    """
    generates a random config by picking parameters from the configuration values
    """
    conf = {}
    for sec in config:
        conf[sec]={}
        for p in config[sec]:
            conf[sec][p] =config[sec][p][np.random.choice(len(config[sec][p]))]
    return conf

def recode_dataframe(df, conf):
    """
    Recodes the values of the dataframe so it can be use with scikit learn
    :param df:
    :param config:
    :return:
    """
    for sec in sorted(conf.keys()):
        for p in sorted(conf[sec].keys()):
            df[p] = df[p].replace(to_replace=[str(v) for v in conf[sec][p]], value=[i for i in range(len(conf[sec][p]))])
    return df

def get_df_configurations(df, conf):
    """
    Generates a set with the configurations already tested in canonical form

    :param df:
    :param config:
    :return:
    """
    sconf = set()

    for i in range(len(df)):
        s = ""
        for sec in sorted(conf.keys()):
            for p in sorted(conf[sec].keys()):
                s+=f"{p}={df.iloc[i][p][0]}/"
            s+='#'
        sconf.add(s)
    return sconf

def config_to_example(conf, confP, vars):
    """
    Transforms a configuration into a recoded example
    :param conf:
    :return:
    """
    a = np.zeros(len(vars))
    for sec in sorted(conf.keys()):
        for p in sorted(conf[sec].keys()):
            a[vars.index(p)] = confP[sec][p].index(conf[sec][p])

    return a.reshape((1,-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_RNN_s2s', help='Experiment configuration')
    parser.add_argument('--pconfig', default='pconfig_RNN_s2s', help='Paramters to explore configuratio')
    parser.add_argument('--test', action='store_true', default=True, help='Print the number of configurations')
    parser.add_argument('--exp',  default='rnns2sactiv4', help='Experiment Name')
    parser.add_argument('--rexp', default='rnns2sactiv4', help='Reference experiment Name')
    parser.add_argument('--npar', type=int, default=10, help='Number of parameter combinations to generate')
    parser.add_argument('--std', type=float, default=0.1, help='Range for the STD to explore')
    parser.add_argument('--confexp', type=int, default=500, help='Number of parameter combinations to explore')
    parser.add_argument('--testdb', action='store_true', default=True, help='Use test database')

    args = parser.parse_args()

    configB = load_config_file(args.config, upload=True)
    configP = load_config_file(args.pconfig, upload=True)

    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    # Find one site 
    sites = list(set([c['site'] for c in col.find({'experiment': args.exp}, ['site'])]))
    # No experiments yet
    if len(sites) ==0:
        # Find one site
        sites = list(set([c['site'] for c in col.find({'experiment': args.rexp}, ['site'])]))
        sites = [sites[0]]
        print(sites)
        lconf = []
        sconf = set()
        i = 0
        while i < args.npar:
            conf = generate_random_config(configP)
            if hash_config(conf) not in sconf:
                lconf.append(conf)
                sconf.add(hash_config(conf))
                i+=1
    # Some results already
    else:

        site = sites[0]
        # Get all unique configurations
        configurations = col.find({'experiment': args.exp, 'site':site})

        # Variables used from the experiments
        arch = list(configP['arch'].keys()) if 'arch' in configP else []
        data = list(configP['data'].keys()) if 'data' in configP else []
        train = list(configP['train'].keys()) if 'train' in configP else []

        lvars = ['hour', 'site', 'test', 'val'] + arch  + data+ train
        ddata = {}
        for var in lvars:
            ddata[var] = []

        # Retrieve all results
        lexp =  col.find({'experiment': args.exp})
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

                for v in arch:
                    lvals.append(str(exp['arch'][v]))

                for v in data:
                    lvals.append(str(exp['data'][v]))

                for v in train:
                    lvals.append(str(exp['training'][v]))

                for var, val in zip(lvars, lvals):
                    ddata[var].append(val)

        # Transform the experiment results into a dataframe and get the experiment mean R^2
        exp_df = pd.DataFrame(ddata)
        exp_df = exp_df.groupby(by=['site']+arch+data + train ,as_index=False).sum()
        exp_df.drop(columns=['hour', 'site'], inplace=True)
        exp_df = exp_df.groupby(by=arch+ data + train ,as_index=False).agg({'test':['mean'],'val':['mean']})

        # Get all the experiments in the dataset in canonical representation
        conf_done = get_df_configurations(exp_df, configP)
        exp_df = recode_dataframe(exp_df, configP)

        # Train a random forest regressor to predict accuracy of new configurations
        dataset = exp_df.to_numpy()
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(dataset[:,:-2], dataset[:,-2])

        max_pred = np.max(dataset[:,-2])
        pred_std = np.std(dataset[:,-2])

        i = 0
        nc = 0
        lconf = []
        print("Scanning configurations ...")
        while i < args.confexp or nc < args.npar:
            conf = generate_random_config(configP)
            if hash_config(conf) not in conf_done:
                v = config_to_example(conf, configP, arch+ data + train)
                pred = rfr.predict(v)
                if pred + (args.std*pred_std) > max_pred:
                    print(conf, pred)
                    lconf.append(conf)
                    nc += 1
                i += 1


    if len(lconf) > 0:

        lsitesconf = []
        for c in lconf:
            lsitesconf +=  change_config(configB, c, sites)

        ids = int(time())
        for n, sc in tqdm(enumerate(lsitesconf)):
        # for n, sc in enumerate(lsitesconf):
            sc['experiment'] = args.exp
            sc['status'] = 'pending'
            sc['result'] = []
            site = sc['data']['datanames'][0].split('-')
            sc['_id'] = f"{ids}{n:05d}{int(site[1]):06d}"
            if not args.test:
                col.insert_one(config)




    # if args.test:
    #     conf = generate_configs(configB)
    #     for c in conf:
    #         print(c)
    # else:
    #     if args.testdb:
    #         mongoconnection = mongolocaltest
    #     client = MongoClient(mongoconnection.server)
    #     db = client[mongoconnection.db]
    #     if mongoconnection.user is not None:
    #         db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    #     col = db[mongoconnection.col]
    #
    #     ids = int(time())
    #     for i, config in tqdm(enumerate(generate_configs(configB))):
    #         config['experiment'] = args.exp
    #         config['status'] = 'pending'
    #         site = config['data']['datanames'][0].split('-')
    #         config['site'] = '-'.join(site[:2])
    #         config['result'] = []
    #         config['_id'] = f"{ids}{i:05d}{int(site[1]):06d}"
    #         col.insert_one(config)
