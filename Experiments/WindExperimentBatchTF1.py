"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""
import argparse
import os
from time import strftime

from Wind.DataBaseConfigurations import getconfig, saveconfig
from Wind.Misc import load_config_file
from Wind2.Train import TrainDispatch, RunConfig

__author__ = 'bejar'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true',
                        default=False)
    parser.add_argument('--info', help="Outputs architecture and training info with results", action='store_true',
                        default=False)
    parser.add_argument('--log', help="Saves output results in the file", default=None)
    parser.add_argument('--gpu', help="Use LSTM/GRU gpu implementation", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--early', help="Early stopping when no improving", action='store_true', default=True)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    parser.add_argument('--proxy', help="Access configurations throught proxy", action='store_true', default=False)
    parser.add_argument('--config', default='config_RNN_dir_reg.json', help='Experiment configuration')
    parser.add_argument('--exp', default=None, help='type of experiment')
    parser.add_argument('--multi', type=int, default=1, help='multi GPU training')
    parser.add_argument('--gpulog', action='store_true', default=False, help='GPU logging')
    parser.add_argument('--mino', action='store_true', default=False, help='Running in minotauro')
    parser.add_argument('--save', action='store_true', default=False, help='Save Model')
    parser.add_argument('--remote', action='store_true', default=False, help='Use remote data')
    parser.add_argument('--secpat', default=None, required=False, type=str,
                        help='Sectiom regexp for retrieving configs')
    parser.add_argument('--dev', default=None, required=False, type=str, help='Select cuda device')
    args = parser.parse_args()

    if not args.gpulog:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if args.dev is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1

    if args.config is None:
        config = getconfig(proxy=args.proxy, mode=args.exp, secpat=args.secpat)
    elif args.mino:
        config = load_config_file(args.config, id=False, mino=True)
    else:
        config = load_config_file(args.config, id=True)

    run_config = RunConfig(impl, verbose, args.tboard, args.best, args.early, args.multi, args.proxy, args.save,
                           args.remote, args.info, args.log)

    dispatch = TrainDispatch()

    if config is not None:

        ############################################
        # Data

        if not 'site' in config:
            site = config['data']['datanames'][0].split('-')
            config['site'] = f"{site[0]}-{site[1]}"
        # try:
        print(f"Running job {config['_id']} {config['site']} {config['arch']['mode']} {strftime('%Y-%m-%d %H:%M:%S')}")
        train_process, architecture = dispatch.dispatch(config['arch']['mode'])

        lresults = train_process(architecture, config, run_config)

        if args.config is None:
            saveconfig(config, lresults, proxy=args.proxy)
        elif args.mino:
            saveconfig(config, lresults, mino=True)
        # else:
        #     for res in lresults:
        #         print(res)
    # except Exception:
    # failconfig(config)
