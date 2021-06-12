#!/usr/bin/python
"""
.. module:: LaunchJobs

LaunchJobs
*************

:Description: LaunchJobs

    Launches new jobs

:Authors: bejar
    

:Version: 

:Created on: 15/02/2019 7:13 

"""

from __future__ import print_function
import argparse
import glob
import os


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', action='store_true', default=False, help='Power9 jobs')
    parser.add_argument('--p', action='store_true', default=False, help='Minotauro jobs')
    parser.add_argument('--test', action='store_true', default=False, help='testing')
    args = parser.parse_args()


    if args.m:
        lfiles = glob.glob('windjobmino*.cmd')

        for j in lfiles:
            if not args.test:
                os.rename(j, 'sub_%s' %j)
                # print('mnsubmit sub_%s' %j)
                # os.system('mnsubmit sub_%s' %j)
                print('sbatch sub_%s' %j)
                os.system('sbatch sub_%s' %j)

            else:
                print('sbatch sub_%s' %j)

    elif args.p:
        lfiles = glob.glob('windjobpower*.cmd')

        for j in lfiles:
            if not args.test:
                os.rename(j, 'sub_%s' %j)
                print('sbatch sub_%s' %j)
                os.system('sbatch sub_%s' %j)
            else:
                print('sbatch sub_%s' %j)
