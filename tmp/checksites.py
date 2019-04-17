"""
.. module:: checksites

checksites
******

:Description: checksites

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  17/04/2019
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

    a = col.find_one(smacexp)

    ssites = set()

    for l in a['sites']:
        for s in l:
            if s in ssites:
                print(f'repe = {s}')
            else:
                ssites.add(s)

