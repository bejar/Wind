"""
.. module:: MeasuresMerge

MeasuresMerge
*************

:Description: MeasuresMerge

    

:Authors: bejar
    

:Version: 

:Created on: 15/09/2020 8:03 

"""


from pymongo import MongoClient
from tqdm import tqdm

from Wind.Private.DBConfig import mongoconnection, mongolocaltest, mongolocalmeasures

__author__ = 'bejar'

if __name__ == '__main__':

    mongomeasures = mongolocalmeasures
    client = MongoClient(mongomeasures.server)
    db = client[mongomeasures.db]
    if mongomeasures.user is not None:
        db.authenticate(mongomeasures.user, password=mongomeasures.passwd)
    col = db[mongomeasures.col]

    mongodata = mongoconnection
    client2 = MongoClient(mongodata.server)
    db = client[mongodata.db]
    if mongoconnection.user is not None:
        db.authenticate(mongodata.user, password=mongodata.passwd)
    col2 = db[mongodata.col]

    configs1 = col.find({'experiment':'measures'})
    configs2 = col2.find({'experiment':'stl'})

    stlres = {}
    for c in configs2:
        stlres[c['site']]= c['result']

    for conf in tqdm(configs1):
        for msr in stlres[conf['site']]:
            conf['result'][msr].update(stlres[conf['site']][msr])
        col.update({'_id': conf['_id']}, {'$set': {'result': conf['result']}})
        # print(conf)
        # if conf['site'] in lsites:
        #     #print(f"REPE {conf['site']}")
        #     nrep += 1
        #     col.delete_one({'_id': conf['_id']})
        # else:
        #     lsites.add(conf['site'])
