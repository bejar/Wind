"""
.. module:: GenerateExpConfPseudoSMAC

GenerateExpConfPseudoSMAC
*************

:Description: GenerateExpConfPseudoSMAC

  Attempt to implement similar SMAC algorithm for meta parameter exploration
    

:Authors: bejar
    

:Version: 

:Created on: 08/04/2019 14:19 

"""



import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
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
        conf['data']['datanames'] = [f'{s}-12']
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
            s += f"{p}={conf[sec][p]}/"
        s += '#'
    return s


def generate_random_config(config):
    """
    generates a random config by picking parameters from the configuration values
    """
    conf = {}
    for sec in config:
        conf[sec] = {}
        for p in config[sec]:
            conf[sec][p] = config[sec][p][np.random.choice(len(config[sec][p]))]
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
            df[p] = df[p].replace(to_replace=[str(v) for v in conf[sec][p]],
                                  value=[i for i in range(len(conf[sec][p]))])
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
                s += f"{p}={df.iloc[i][p][0]}/"
            s += '#'
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

    return a.reshape((1, -1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_RNN_s2s', help='Experiment configuration')
    parser.add_argument('--pconfig', default='pconfig_RNN_s2s', help='Paramters to explore configuration')

    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')

    parser.add_argument('--exp', default='rnns2sactiv4', help='Experiment Name')

    parser.add_argument('--npar', type=int, default=10, help='Number of parameter combinations to generate')
    parser.add_argument('--std', type=float, default=.4, help='Range for the STD of the best prediction to explore')

    parser.add_argument('--confexp', type=int, default=2000, help='Number of parameter combinations to explore')

    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    # Strategies/Stages
    parser.add_argument('--init', type=int, default=None, help='Initialize the random sites for exploration')
    parser.add_argument('--refexp', default='rnns2sactiv4', help='Use reference experiment (Name')

    parser.add_argument('--rand', action='store_true', default=False, help='Generate random configurations')


    args = parser.parse_args()

    # Template configuration file
    configB = load_config_file(args.config, upload=True)
    # Parameters space configuration file (all parameters and values to explore)
    configP = load_config_file(args.pconfig, upload=True)

    # DB stuff
    if args.testdb:
        mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]


    ## Pick a random sample of sites and divide it in exploration stages
    ## Default is 1000 divided in batches of 25
    if args.init is not None:
        smacexp = col.find_one({'SMAC':'init', 'experiment': args.exp}, ['site'])
        if smacexp:
            raise NameError("Experiment already initialized")
        else:
            lsites = np.choose(range())



    refexp = False
    # Find one site
    sites = list(set([c['site'] for c in col.find({'experiment': args.exp}, ['site'])]))
    # No experiments yet
    if len(sites) == 0:
        if (args.refexp is not None):
            refexp = True
            # Find one site from the reference experiment
            sites = list(set([c['site'] for c in col.find({'experiment': args.refexp}, ['site'])]))
        else:
            raise NameError("No sites in the experiments and no reference experiment")

    print(sites)


