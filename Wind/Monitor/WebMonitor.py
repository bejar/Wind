"""
.. module:: WebMonitor

WebMonitor
******

:Description: WebMonitor

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  18/03/2018
"""

import socket

from flask import Flask, render_template, request
from pymongo import MongoClient
from Wind.Private.DBConfig import mongoconnection

__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 9000

app = Flask(__name__)


@app.route('/Monitor')
def info():
    """
    Status de las ciudades
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    exp = col.find({'status': 'done'})
    done = len([v for v in exp])


    exp = col.find({'status': 'working'})
    work = [v['btime'] for v in exp]

    exp = col.find({'status': 'pending'})
    pend= len([v for v in exp])

    return render_template('Monitor.html', done=done, pend=pend, work=work)