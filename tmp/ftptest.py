"""
.. module:: ftptest

ftptest
*************

:Description: ftptest

    

:Authors: bejar
    

:Version: 

:Created on: 19/07/2018 10:03 

"""

__author__ = 'bejar'



import pysftp
from Wind.Config import wind_data_path

if __name__ == '__main__':
    srv = pysftp.Connection(host="polaris.cs.upc.edu", username='bejar', private_key='~/.ssh/authorized_keys')
    srv.get(wind_data_path + '0-0-12.py', wind_data_path)