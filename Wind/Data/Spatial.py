"""
.. module:: Spatial

Spatial
*************

:Description: Spatial

    Experiments with geographical distribution of siter

:Authors: bejar
    

:Version: 

:Created on: 08/10/2018 9:37 

"""

__author__ = 'bejar'


import numpy as np
from sklearn.neighbors import KDTree
from Wind.Config.Paths import wind_data_path
from Wind.Misc import find_exp, count_exp, sel_result

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo= False
else:
    _has_mongo = True

if __name__ == '__main__':

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    if mongoconnection.passwd is not None:
        db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]


    coords = np.load(wind_data_path +'/coords.npy')
    memo = np.zeros(coords.shape[0])
    query = {"experiment": "rnnseq2seq", "site": ""}
    tree = KDTree(coords, leaf_size=1)
    count = 0
    for i in range(coords.shape[0]):
        # print(i, end=' ')
        # if i%100 == 0:
        #     print()
        # dist= tree.query_radius(coords[0,:].reshape(1, -1), r=0.1, return_distance=True, sort_results=True)
        dist= tree.query_radius(coords[i,:].reshape(1, -1), r=0.05, count_only=False, return_distance=False)[0]
        # print(i, dist)
        try:
            if len(dist) > 1:
                tsum = 0
                for j in dist:
                    if j!= i:
                        if memo[j] == 0:
                            query['site'] = '%d-%d'% (j/500,j)
                            exp = col.find_one(query)
                            data = np.array(exp['result'])
                            vsum = np.sum(data[:,1])
                            tsum += vsum
                            memo[j] = vsum
                        else:
                            tsum += memo[j]

                tmean = tsum/(len(dist)-1)
                query['site'] = '%d-%d'% (i/500,i)
                exp = col.find_one(query)
                data = np.array(exp['result'])
                vsum = np.sum(data[:,1])
                if vsum < (tmean *0.9):
                    count +=1
                    # print()
                    print('***')
                    print(i, vsum, tmean, len(dist))
                    col.update({'_id': exp['_id']}, {'$set': {'status': 'pending'}})


        except Exception:
            print('!',j,'!')
            # print(i)
            pass
    print('SUSP=',count)
