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
    print('%d Configurations' % len(lconf))
    return lconf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configbatchregdir', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    args = parser.parse_args()

    configB = load_config_file(args.config)

    if args.test:
        len(generate_configs(configB))
    else:
        client = MongoClient(mongoconnection.server)
        db = client[mongoconnection.db]
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
        col = db[mongoconnection.col]

        ids = int(time())
        for i, config in enumerate(generate_configs(configB)):
            config['status'] = 'pending'
            config['result'] = []
            config['_id'] = str(ids + i)
            col.insert(config)

