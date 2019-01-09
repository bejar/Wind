"""
.. module:: GenerateExpConf

GenerateExpConf
*************

:Description: GenerateExpConf

    The configuration file has the usual format, the script asks for the values to change from the file

    Uploads to the DB the cartesian product of all the values of the lists parameters

    WARNING!!! use the test flag to know how many configurations are generated

:Authors: bejar
    

:Version: 

:Created on: 16/03/2018 12:29 

"""

import argparse
from time import time

from Wind.Misc import load_config_file
from Wind.Private.DBConfig import mongolocaltest, mongoconnection
from copy import deepcopy
from pymongo import MongoClient
import numpy as np
import json
from itertools import combinations
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


def getinput(field, t):
    """
    get input and checks that has the correct type
    :param t:
    :return:
    """
    ok = False
    while not ok:
        resp = input(f"Type the list of values to use for [{field}] as a string: ")
        res = json.loads(resp)
        if type(res) != list:
            print("It is not a list")
        else:
            ok = True
            for v in res:
                ok = ok and type(v) == t
            if not ok:
                print("Some of the values have not the correct type")

    return res


def ask_integer(question):
    """
    asks for an integer value
    :return:
    """
    ok = False
    while not ok:
        v = input(question)
        try:
            v = int(v)
            ok = True
        except ValueError:
            ok = False
    return v


def ask_config(config):
    """
    Asks for the fields of the configuration

    :param config:
    :return:
    """
    outconfig = {'data':{}, 'arch':{}, 'training':{}}

    print("- Data section:")
    print("---------------")
    for v in config['data']:
        print(f"# {v} ({config['data'][v]}):")
        if v not in ['datanames', 'vars']:
            if input("Change (y/N)? ") in ['y', 'yes']:
                outconfig['data'][v] = getinput(v, type(config['data'][v]))
            else:
                outconfig['data'][v] = [config['data'][v]]
        else:
            if v == 'datanames':
                if input("Change datanames (y/N)? ") not in ['y', 'yes']:
                    outconfig['data'][v] = [config['data'][v]]
                elif input("Random datanames (y/N)? ") in ['y', 'yes']:
                    n = ask_integer("How many? ")
                    lsites = np.array([i for i in range(126000)])
                    lsites = np.random.choice(lsites, n, replace=False)
                    outconfig['data'][v] = [[f"{site // 500}-{site}-12"] for site in lsites]
                elif input("Other experiment sites (y/N)? ") in ['y', 'yes']:
                    if args.testdb:
                        mongoconnection = mongolocaltest
                    client = MongoClient(mongoconnection.server)
                    db = client[mongoconnection.db]
                    if mongoconnection.user is not None:
                        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
                    col = db[mongoconnection.col]

                    oexp = input("Experiment? ")
                    resp = col.find({'experiment': oexp}, ['site'])
                    sites = list(np.unique([f"{site['site']}-12" for site in resp]))
                    if len(sites) == 0:
                        raise NameError("No sites for that experiment")
                    else:
                        outconfig['data'][v] = [[s] for s in sites]
                else:
                    outconfig['data'][v] = [[f"{site // 500}-{site}-12"] for site in getinput(v, int)]
            elif v == 'vars':
                if input("Change vars (y/N)? ") not in ['y', 'yes']:
                    outconfig['data'][v] = [config['data'][v]]
                elif (input("all subsets (y/N)? ")) in ['y', 'yes']:
                    comb = []
                    for c in range(1, len(config['data'][v])):
                        comb.extend([list(c) for c in combinations(config['data'][v][1:], c)])
                    outconfig['data'][v] = [[config['data'][v][0]] + c for c in comb]

    print()
    print("- Architecture section:")
    print("-----------------------")
    for v in config['arch']:
        print(f"# {v} ({config['arch'][v]}):")
        if v != 'mode':
            if (input("Change (y/N)? ")) in ['y', 'yes']:
                outconfig['arch'][v] = getinput(v, type(config['arch'][v]))
            else:
                outconfig['arch'][v] = [config['arch'][v]]
        else:
            outconfig['arch'][v] = [config['arch'][v]]

    print()
    print("- Training section:")
    print("-------------------")
    for v in config['training']:
        print(f"# {v} ({config['training'][v]}):")
        if (input("Change (y/N)? ")) in ['y', 'yes']:
            outconfig['training'][v] = getinput(v, type(config['training'][v]))
        else:
            outconfig['training'][v] = [config['training'][v]]
    return outconfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_MLP_s2s', help='Experiment configuration')
    parser.add_argument('--test', action='store_true', default=False, help='Print the number of configurations')
    parser.add_argument('--testdb', action='store_true', default=False, help='Use test database')
    parser.add_argument('--exp', required=True, help='Experiment Name')

    args = parser.parse_args()

    configB = load_config_file(args.config, upload=True)
    configB = ask_config(configB)


    if args.test:
        conf = generate_configs(configB)
        print(conf[0])
        print(len(conf))
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
            config['result'] = []
            site = config['data']['datanames'][0].split('-')
            config['site'] = '-'.join(site[:2])
            config['_id'] = f"{ids}{i:05d}{int(site[1]):06d}"
            col.insert_one(config)
