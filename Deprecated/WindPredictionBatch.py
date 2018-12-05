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

from Wind.Miscelanea import load_config_file
from Wind.DataBaseConfigurations import getconfig, saveconfig, failconfig
from Deprecated import train_dirregression_architecture, train_seq2seq_architecture, train_MLP_regs2s_architecture, \
    train_ensemble_architecture, train_convdirregression_architecture, train_MLP_dirreg_architecture, \
    train_svm_dirregression_architecture, train_convo_regs2s_architecture, train_persistence, \
    train_seq2seqatt_architecture
from Wind.Config.Paths import remote_wind_data_path
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
    parser.add_argument('--remote', action='store_true', default=False, help='Use remote data')
    parser.add_argument('--secpat', default=None, required=False, type=str, help='Sectiom regexp for retrieving configs')
    parser.add_argument('--dev', default=None, required=False, type=str, help='Select cuda device')
    args = parser.parse_args()

    if not args.gpulog:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if args.dev is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    verbose = 1 if args.verbose else 0
    impl = 1 if args.gpu else 2

    if args.config is None:
        config = getconfig(proxy=args.proxy, mode=args.exp, secpat=args.secpat)
    elif args.mino:
        config = load_config_file(args.config, id=False, mino=True)
    else:
        config = load_config_file(args.config, id=True)

    if config is not None:

        # print(config)

        try:
            print('Running job %s %s %s' % (config['data']['datanames'][0], config['arch']['mode'], strftime('%Y-%m-%d %H:%M:%S')))

            if config['arch']['mode'] == 'regdir':
                lresults = train_dirregression_architecture(config, impl, verbose, args.tboard, args.best, args.early,
                                                            multi=args.multi, proxy=args.proxy, save=args.save,
                                                            remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'seq2seq':
                lresults = train_seq2seq_architecture(config, impl, verbose, args.tboard, args.best, args.early,
                                                      multi=args.multi, save=args.save,
                                                      remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'seq2seqa':
                lresults = train_seq2seqatt_architecture(config, impl, verbose, args.tboard, args.best, args.early,
                                                         multi=args.multi, save=args.save,
                                                         remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'mlps2s':
                lresults = train_MLP_regs2s_architecture(config, verbose, args.tboard, args.best, args.early,
                                                         multi=args.multi, save=args.save,
                                                         remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'mlpdir':
                lresults = train_MLP_dirreg_architecture(config, verbose, args.tboard, args.best, args.early,
                                                         multi=args.multi, save=args.save,
                                                         remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'convo':
                lresults = train_convdirregression_architecture(config, verbose, args.tboard, args.best, args.early,
                                                                multi=args.multi, save=args.save,
                                                                remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'convos2s':
                lresults = train_convo_regs2s_architecture(config, verbose, args.tboard, args.best, args.early,
                                                           multi=args.multi, save=args.save,
                                                           remote=remote_wind_data_path if args.remote else None)
            elif 'ens' in config['arch']['mode']:
                lresults = train_ensemble_architecture(config, verbose, args.tboard, args.best, args.early,
                                                       multi=args.multi,
                                                       remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'svmdir':
                lresults = train_svm_dirregression_architecture(config, verbose,
                                                                remote=remote_wind_data_path if args.remote else None)
            elif config['arch']['mode'] == 'persistence':
                lresults = train_persistence(config, verbose, remote=remote_wind_data_path if args.remote else None)

            if args.config is None:
                saveconfig(config, lresults, proxy=args.proxy)
            elif args.mino:
                saveconfig(config, lresults, mino=True)
            else:
                for res in lresults:
                    print(res)
        except Exception as e:
            print(e)
            if not args.mino:
                failconfig(config, proxy=args.proxy)
