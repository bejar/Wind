"""
.. module:: MapExperiment

GenerateExpConf
*************

:Description: MapExperiment

    Generates two HTML maps for the test and validation results of an experiment

:Authors: bejar


:Version:

:Created on: 16/03/2018 12:29

"""

from Wind.Results import DBResults
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='Persistence', help='experiment')
    parser.add_argument('--exp2', default=None, help='experiment')
    parser.add_argument('--sample', default='Persistence', help='experiment')
    args = parser.parse_args()


    if args.exp2 is None:
        query = {'status': 'done', "experiment": args.exp}
        results = DBResults()
        results.retrieve_results(query)
        if results.size() > 0:
            results.sample(args.sample)
            results.plot_map(mapbox=True,dset=('val'))

