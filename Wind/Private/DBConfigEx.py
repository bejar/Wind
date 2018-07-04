"""
.. module:: DBConfig

DBConfig
*************

:Description: DBConfig



:Authors: bejar


:Version:

:Created on: 15/02/2017 9:21

"""

__author__ = 'bejar'


class MongoData:
    def __init__(self, server, db, user, passwd, collect):
        self.server = server
        self.db = db
        self.user = user
        self.passwd = passwd
        self.col = collect


mongoconnection = MongoData('mongodb://localhost/', 'Database', 'User', 'Password',
                            'Collection')
