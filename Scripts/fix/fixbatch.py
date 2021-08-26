"""
.. module:: fixbatch.py

fixbatch.py
******

:Description: fixbatch.py

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  26/08/2021
"""

__author__ = 'bejar'


import argparse
import json
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int,  help='batchsize')

    args = parser.parse_args()
    files = glob.glob('*.work')

    for f in files:
        fp = open(f, 'r')

        s = ''

        for l in fp:
            s += l

        config = json.loads(s)

        config['training']['batch'] = args.batch

        sconf = json.dumps(config, indent=4, sort_keys=True)

        fconf = open(f.replace(".work", ".json"), 'w')
        fconf.write(sconf + '\n')
        fconf.close()

        os.remove(f)