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
    args = parser.parse_args()

    query = {'status': 'done', "experiment": args.exp, "site": {"$regex": "."}}
    results = DBResults()
    results.retrieve_results(query)
    if results.size() > 0:
        # results.sample(0.1)
        # results.select_best_worst_sum_accuracy()
        results.map_results()

