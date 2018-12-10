"""
.. module:: DBResults

DBResults
*************

:Description: DBResults

    Access to the results in a mongoDB

:Authors: bejar
    

:Version: 

:Created on: 10/12/2018 15:09 

"""
import numpy as np

try:
    from pymongo import MongoClient
    from Wind.Private.DBConfig import mongoconnection
except ImportError:
    _has_mongo = False
else:
    _has_mongo = True

__author__ = 'bejar'

class DBResults:
    """
    Access to results in a mongoDB
    """

    connection = None
    client = None
    user = None
    password = None
    db = None
    col = None

    def __init__(self, conn=mongoconnection, test=""):
        """
        configures the DB
        :param DB:
        """
        self.connection = conn
        self.client = MongoClient(conn.server)
        self.db = self.client[conn.db]
        if conn.passwd is not None:
            self.db.authenticate(conn.user, password=conn.passwd)
        self.col = self.db[conn.col + test]

    def find_exp(self, query):
        """
        Returns all the experiments in the DB that match the query

        :param query:
        :return:
        """
        return self.col.find(query)

    def count_exp(self, query):
        """
        Counts how many experiments in the DB match the query

        :param query:
        :return:
        """
        return self.col.count(query)

    def sel_result(self, query, ncol):
        """
        Selects from a list of configurations with results the result in the column 1 (test) or 2 (validation)

        :param query:
        :param lexp:
        :param ncol:
        :return:
        """
        lexp = self.find_exp(query)
        ldata = []
        for exp in lexp:

            # To maintain backwards compatibility
            if 'result' in exp:
                data = np.array(exp['result'])
            elif 'results' in exp:
                data = np.array(exp['results'])

            ldata.append((int(exp['data']['datanames'][0].split('-')[1]), data[:, ncol]))
        ldata = sorted(ldata, key=lambda x: x[0])

        return np.array([v[0] for v in ldata]), np.array([v[1] for v in ldata])

    def sel_upper_lower(self, exp, mode, column, upper=100, lower=100):
        """
        Selects a number of experiment with the best and worst accuracy

        column selects test (1) and validation (2)
        :param exp:
        :param mode:
        :param upper:
        :param lower:
        :return:
        """
        exps = self.col.find({'experiment': exp, 'arch.mode': mode})

        lexps = []
        for e in exps:
            if 'result' in e:
                lexps.append((np.sum(np.array(e['result'])[:, column]), e['site'], np.array(e['result'])[:, column]))
            elif 'results' in e:
                lexps.append((np.sum(np.array(e['results'])[:, column]), e['site'], np.array(e['results'])[:, column]))

        lupper = [(v.split('-')[1], s) for _, v, s in sorted(lexps, reverse=True)][:upper]
        llower = [(v.split('-')[1], s) for _, v, s in sorted(lexps, reverse=False)][:lower]
        lexps = []
        lexps.extend(lupper)
        lexps.extend(llower)

        return np.array([v[0] for v in lexps]), np.array([v[1] for v in lexps])
