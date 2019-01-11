"""
.. module:: DBConfig

DBConfig
*************

:Description: DBConfig

  Use this as example to generate the sc

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


    def __repr__(self):
        print(self.server, self.db, self.col)


mongoconnection = MongoData('mongodb://localhost:27017/', 'Database', 'User', 'Password',
                            'Collection')

# Mongo local connection
mongolocal = MongoData('mongodb://localhost:27017/', 'Database', 'User', 'Password',
                            'Collection')

# Token for mapbox.com to be able to plot maps with geographical information with plotly
mapbox_token = ""