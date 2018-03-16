"""
.. module:: GenerateExpConf

GenerateExpConf
*************

:Description: GenerateExpConf

    

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 12:29 

"""

import argparse
from time import time

from Wind.Util import load_config_file
from Wind.Data import generate_dataset
from Wind.Private.DBConfig import mongoconnection
from copy import deepcopy
from pymongo import MongoClient

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

    return lconf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configBatch1', help='Experiment configuration')
    args = parser.parse_args()

    configB = load_config_file(args.config)

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    for config in generate_configs(configB):
        config['status'] = 'pending'
        config['result'] = []
        col.insert(config)

