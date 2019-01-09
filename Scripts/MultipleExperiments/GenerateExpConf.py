"""
.. module:: GenerateExpConf

GenerateExpConf
*************

:Description: GenerateExpConf

    The configuration file has lists of values for all the parameters instead of just the value

    Uploads to the DB the cartesian product of all the values of the lists parameters

    WARNING!!! use the test flag to know how many configurations are generated

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

__author__ = 'bejar'


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
    parser.add_argument('--config', default='Mconfig_MLP_s2s', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--exp', required=True, help='Experiment Name')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')

    args = parser.parse_args()

    configB = load_config_file(args.config, upload=True)

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
