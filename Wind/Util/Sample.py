"""
.. module:: Sample

Sample
*************

:Description: Sample

    Functions for Generating samples of sites

:Authors: bejar
    

:Version: 

:Created on: 18/09/2019 8:23 

"""

import numpy as np
from pymongo import MongoClient
from Wind.Private.DBConfig import mongoconnection, mongolocaltest

__author__ = 'bejar'


def uniform_sample(nsamples, batch=None):
    """
    Generates a sample of sites using a uniform distribution
    Returs the list of sites or a list of site batches if batch is not None
    :param nsamples:
    :param batch:
    :param store:
    :return:
    """
    lsites = np.random.choice(range(126692), nsamples, replace=False)
    lsites = [f'{site // 500}-{site}' for site in lsites]

    if batch is not None:
        lbatches = []
        for i in range(0, len(lsites), batch):
            lbatches.append(lsites[i:i + batch])
        lsites = lbatches

    return lsites

def entropy_sample(nsamples, batch=None, bucket=10):
    """
    Generates a sample of sites using the distribution of the spectral entropy of sites' wind
    :param nsamples:
    :param batch:
    :return:
    """
    mongoconnection = mongolocaltest
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    measures = col.find({'experiment': 'measures'})
    ames = []
    for m in measures:
        ames.append((m['result']['wind_speed']['SpecEnt'], m['site']))
    ames = sorted(ames)
    lsites = []
    lbbucket = 126692//bucket
    for i in range(bucket):
        sites = np.random.choice(range(lbucket*i, lbucket*(i+1)), nsamples//bucket, replace=False)
        lsites.extend([ames[v][1]  for v in sites])

    np.random.shuffle(lsites)

    if batch is not None:
        lbatches = []
        for i in range(0, len(lsites), batch):
            lbatches.append(lsites[i:i + batch])
        lsites = lbatches

    return lsites


if __name__ == '__main__':
    lsites = entropy_sample(200)
    print(lsites)

