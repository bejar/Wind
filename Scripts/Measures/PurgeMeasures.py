"""
.. module:: PurgeMeasures

TransformData
*************

:Description: Deletes duplicated entries


:Authors: bejar
    

:Version: 

:Created on: 12/04/2018 10:21 

"""


import argparse
from Wind.Private.DBConfig import mongoconnection, mongolocaltest
from pymongo import MongoClient

from tqdm import tqdm

__author__ = 'bejar'

__author__ = 'bejar'

if __name__ == '__main__':
    mongoconnection = mongolocaltest

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.user is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    
    configs = col.find({'experiment':'measures'})
    lsites = set()
    nrep = 0 
    for conf in tqdm(configs):
        if conf['site'] in lsites:
            #print(f"REPE {conf['site']}")
            nrep += 1
            col.delete_one({'_id': conf['_id']})
        else:
            lsites.add(conf['site'])
    print(nrep)
#    for conf in tqdm(configs):
#        lsites.add(conf['site'])
#    for s in range(126691):
#        if f"{s//500}-{s}" not in lsites:
#            print(s)




