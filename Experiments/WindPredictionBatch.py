"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

from __future__ import print_function

from time import strftime

from Wind.Util import load_config_file
from Wind.Training import getconfig, saveconfig, failconfig
from Wind.Models import train_dirregression_architecture, train_seq2seq_architecture, train_MLP_regs2s_architecture,\
    train_ensemble_architecture, train_convdirregression_architecture, train_MLP_dirreg_architecture, \
    train_svm_dirregression_architecture, train_convo_regs2s_architecture, train_persistence

import os
import argparse


__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--early', help="Early stopping when no improving", action='store_true', default=True)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    parser.add_argument('--proxy', help="Access configurations throught proxy", action='store_true', default=False)
    parser.add_argument('--config', default=None, help='Experiment configuration')
    parser.add_argument('--exp', default=None, help='type of experiment')
    parser.add_argument('--multi', type=int, default=1, help='multi GPU training')
    parser.add_argument('--gpulog', action='store_true', default=False, help='GPU logging')
    parser.add_argument('--mino', action='store_true', default=False, help='Running in minotauro')
    parser.add_argument('--save', action='store_true', default=False, help='Save Model')
    args = parser.parse_args()

    if not args.gpulog:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1

    if args.config is None:
        config = getconfig(proxy=args.proxy, mode=args.exp)
    elif args.mino:
        config = load_config_file(args.config, id=False)
    else:
        config = load_config_file(args.config, id=True)

    if config is not None:

        ############################################
        # Data

        # try:
            print('Running job %s %s %s' % (config['_id'], config['arch']['mode'], strftime('%Y-%m-%d %H:%M:%S')))

            if config['arch']['mode'] == 'regdir':
                lresults = train_dirregression_architecture(config, impl, verbose, args.tboard, args.best, args.early, multi=args.multi, proxy=args.proxy, save=args.save)
            elif config['arch']['mode'] == 'seq2seq':
                lresults = train_seq2seq_architecture(config, impl, verbose, args.tboard, args.best, args.early, multi=args.multi, save=args.save)
            elif config['arch']['mode'] == 'mlps2s':
                lresults = train_MLP_regs2s_architecture(config, verbose, args.tboard, args.best, args.early, multi=args.multi, save=args.save)
            elif config['arch']['mode'] == 'mlpdir':
                lresults = train_MLP_dirreg_architecture(config, verbose, args.tboard, args.best, args.early, multi=args.multi, save=args.save)
            elif config['arch']['mode'] == 'convo':
                lresults = train_convdirregression_architecture(config, verbose, args.tboard, args.best, args.early, multi=args.multi, save=args.save)
            elif config['arch']['mode'] == 'convos2s':
                lresults = train_convo_regs2s_architecture(config, verbose, args.tboard, args.best, args.early, multi=args.multi, save=args.save)
            elif 'ens' in config['arch']['mode']:
                lresults = train_ensemble_architecture(config, verbose, args.tboard, args.best, args.early, multi=args.multi)
            elif config['arch']['mode'] == 'svmdir':
                lresults = train_svm_dirregression_architecture(config, verbose)
            elif config['arch']['mode'] == 'persistence':
                lresults = train_persistence(config, verbose)

            if args.config is None:
                saveconfig(config, lresults, proxy=args.proxy)
            elif args.mino:
                saveconfig(config, lresults, mino=True)
            else:
                for res in lresults:
                    print(res)
        # except Exception:
            # failconfig(config)

