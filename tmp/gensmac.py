"""
.. module:: cpsmac

cpsmac
*************

:Description: cpsmac

 Generate SMAC item in database from old experiments

:Authors: bejar
    

:Version: 

:Created on: 08/04/2019 16:34 

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


if __name__ == '__main__':
    # DB stuff

    mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    maxsites = 1500
    BATCH = 25
    # Check if experiment already exists
    smacexp = {'SMAC':'init', 'smexperiment': 'rnns2sactiv4'}

    lexsites = list(set([c['site'] for c in col.find({'experiment': 'rnns2sactiv4'}, ['site'])]))
    print(lexsites)
    ssites = set(lexsites)

    lsites = np.random.choice(range(126691), 3000, replace=False)
    lsites = [f'{site//500}-{site}' for site in lsites]

    for s in lsites:
        if s not in ssites:
            lexsites.append(s)
        if len(lexsites) == maxsites:
            break

    lbatches = []
    for i in range(0,len(lexsites), BATCH):
        lbatches.append(lexsites[i:i+BATCH])
    smacexp['sites'] = lbatches
    smacexp['batch'] = BATCH

    print(smacexp)
    col.insert_one(smacexp)

