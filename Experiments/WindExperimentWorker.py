"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindExperimentWorker

:Authors: bejar

    Takes configurations from jobs directory

:Version: 

:Created on: 06/09/2017 9:47 

"""
import argparse
import logging
import os
import sys
import warnings
from glob import glob
from time import strftime, sleep

from Wind.Config.Paths import wind_jobs_path, wind_local_jobs_path
from Wind.DataBaseConfigurations import saveconfig
from Wind.Misc import load_config_file
from Wind.Train import TrainDispatch, RunConfig

logging.getLogger('tensorflow').disabled = True
# logging.getLogger("tensorflow").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

__TF2__ = True

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
    parser.add_argument('--jobsdir', help="Worker jobs directory", default="")
    parser.add_argument('--exp', default=None, help='type of experiment')
    parser.add_argument('--multi', type=int, default=1, help='multi GPU training')
    parser.add_argument('--gpulog', action='store_true', default=False, help='GPU logging')
    parser.add_argument('--mino', action='store_true', default=False, help='Running in minotauro')
    parser.add_argument('--local', action='store_true', default=False, help='Running local jobs')
    parser.add_argument('--save', action='store_true', default=False, help='Save Model')
    parser.add_argument('--remote', action='store_true', default=False, help='Use remote data')
    parser.add_argument('--secpat', default=None, required=False, type=str,
                        help='Sectiom regexp for retrieving configs')
    parser.add_argument('--dev', default=None, required=False, type=str, help='Select cuda device')
    parser.add_argument('--nit', type=int, default=1, help='iterate n times')

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if not args.gpulog:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if args.dev is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 1

    if args.mino:
        pre = wind_jobs_path
    elif args.local:
        pre = wind_local_jobs_path

    if not os.path.exists(f'{pre}/{args.jobsdir}'):
        os.mkdir(f'{pre}/{args.jobsdir}')

    for i in range(args.nit):
        config = None
        nl = 0
        while config is None:
            lfiles = glob(f'{pre}/{args.jobsdir}/*.json')
            if len(lfiles) == 0:
                lend = glob(f'{pre}/{args.jobsdir}/.end')
                if len(lend) != 0:
                    os.remove(f'{pre}/{args.jobsdir}/.end')
                    lend = glob(f'{pre}/{args.jobsdir}/*.done')
                    for fdone in lend:
                        os.remove(fdone)
                    os.rmdir(f'{pre}/{args.jobsdir}')
                    sys.exit(1)
                else:
                    sleep(30)
                    if nl > 10:
                        sys.exit(1)
                    nl += 1
            else:
                config = lfiles[0].split('/')[-1].split('.')[0]

        if args.mino:
            config = load_config_file(args.jobsdir + '/' + config, id=False, mino=True)
        elif args.local:
            config = load_config_file(args.jobsdir + '/' + config, id=False, local=True)
        else:
            config = load_config_file(args.jobsdir + '/' + config, id=True)

        run_config = RunConfig(impl, verbose, args.tboard, args.best, args.early, args.multi, args.proxy, args.save,
                               args.remote, args.info, args.log)

        dispatch = TrainDispatch()

        if config is not None:

            ############################################
            # Data

            if not 'site' in config:
                site = config['data']['datanames'][0].split('-')
                config['site'] = f"{site[0]}-{site[1]}"

            print(f"Running job {config['_id']} {config['site']} {config['arch']['mode']} {strftime('%Y-%m-%d %H:%M:%S')}")
            train_process, architecture = dispatch.dispatch(config['arch']['mode'])

            lresults = train_process(architecture, config, run_config)

            if config is None:
                saveconfig(config, lresults, proxy=args.proxy)
            elif args.mino:
                saveconfig(config, lresults, mino=True)
            elif args.local:
                saveconfig(config, lresults, local=True)

    sys.exit(0)