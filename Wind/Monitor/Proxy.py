"""
.. module:: Proxy

Proxy
*************

:Description: Proxy

    

:Authors: bejar
    

:Version: 

:Created on: 19/03/2018 6:57 

"""
import socket

from flask import Flask, render_template, request
from pymongo import MongoClient
from Wind.Private.DBConfig import mongoconnection

__author__ = 'bejar'


# Configuration stuff
hostname = socket.gethostname()
port = 9001

app = Flask(__name__)
