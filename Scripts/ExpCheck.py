"""
.. module:: ExpCheck

ExpCheck
*************

:Description: ExpCheck

    

:Authors: bejar
    

:Version: 

:Created on: 03/09/2018 8:50 

"""
from Wind.Private.DBConfig import mongoconnection
from pymongo import MongoClient

__author__ = 'bejar'


if __name__ == '__main__':
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    total = 0
    for i in range(0,254):
        configs = col.find({'experiment':'rnnseq2seq', 'status':'pending', 'site': {'$regex':'^%s-.'%str(i)}})
        count = 0
        lsites = []
        for conf in configs:
            count += 1
            lsites.append(conf['site'])
            # if conf['site'] in lsites:
            #     print(conf['_id'])
            #     col.remove({'_id':conf['_id']})
            # print(conf['site'])
#        for l in sorted(lsites):
#            print(l)
        total += count
        print (i, count)
    print(total)
