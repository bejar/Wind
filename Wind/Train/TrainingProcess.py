"""
.. module:: TrainingProcess

TrainingProcess
*************

:Description: TrainingProcess

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 7:53 

"""

try:
    from keras.layers import CuDNNGRU, CuDNNLSTM
except ImportError:
    _has_CuDNN = False
else:
    _has_CuDNN = True

try:
    from keras.utils import multi_gpu_model
except ImportError:
    _has_multigpu = False
else:
    _has_multigpu = True

import tensorflow as tf
from Wind.Data.DataSet import Dataset
from Wind.Config import wind_data_path
from time import strftime
from Wind.Training import updateprocess

__author__ = 'bejar'


def train_dirregression(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    for iter in range(niter):

        for ahead in range(iahead, sahead + 1):

            if runconfig.verbose:
                print('-----------------------------------------------------------------------------')
                print(f"Steps Ahead = {ahead}")

            dataset = Dataset(config=config['data'], data_path=wind_data_path)
            dataset.generate_dataset(ahead=[ahead,ahead], mode=architecture.data_mode, remote=runconfig.remote)

            ############################################
            # Model

            config['idimensions'] = dataset.train_x.shape[1:]

            arch = architecture(config, runconfig)

            if runconfig.multi == 1:
                arch.generate_model()
            else:
                with tf.device('/cpu:0'):
                    arch.generate_model()

            if runconfig.verbose:
                arch.summary()
                arch.plot()
                dataset.summary()
                print()

            ############################################
            # Training
            arch.train(dataset.train_x, dataset.train_y, dataset.val_x, dataset.val_y)

            ############################################
            # Results

            r2val, r2test = arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)
            lresults.append((ahead, r2val, r2test))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            # Update result in db
            if config is not None and not runconfig.proxy:
                from Wind.Training import updateprocess
                updateprocess(config, ahead)

            arch.save('-A%d-R%02d' % (ahead, iter))
            del dataset

    arch.log_result(lresults)

    return lresults


def train_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """
    ahead = config['data']['ahead']

    if not type(ahead) == list:
        ahead = [1, ahead]

    dataset = Dataset(config=config['data'], data_path=wind_data_path)
    dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] + 1
    else:
        odimensions = ahead

    lresults = []
    for iter in range(niter):

        config['idimensions'] = dataset.train_x.shape[1:]
        config['odimensions'] = odimensions
        arch = architecture(config, runconfig)

        if runconfig.multi == 1:
            arch.generate_model()
        else:
            with tf.device('/cpu:0'):
                arch.generate_model()

        if runconfig.verbose:
            arch.summary()
            arch.plot()
            dataset.summary()
            print()

        ############################################
        # Training
        arch.train(dataset.train_x, dataset.train_y, dataset.val_x, dataset.val_y)

        ############################################
        # Results

        lresults.extend(arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save('-A%d-%d-R%02d' % (ahead[0], ahead[1], iter))

    arch.log_result(lresults)

    return lresults


def train_persistence(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    for ahead in range(iahead, sahead + 1):

        if runconfig.verbose:
            print("-----------------------------------------------------------------------------")
            print(f"Steps Ahead = {ahead}")

        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=[ahead,ahead], mode=architecture.data_mode, remote=runconfig.remote)

        arch = architecture(config, runconfig)

        if runconfig.verbose:
            dataset.summary()

        val_r2, test_r2 = arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)
        lresults.append((ahead, val_r2, test_r2))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

        del dataset

    arch.log_result(lresults)
    return lresults


def train_svm_dirregression(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    lresults = []
    for ahead in range(iahead, sahead + 1):

        if runconfig.verbose:
            print('-----------------------------------------------------------------------------')
            print(f'Steps Ahead = {ahead} ')

        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)

        ############################################
        # Model

        arch = architecture(config, runconfig)

        if runconfig.verbose:
            arch.summary()
            dataset.summary()

            print()

        ############################################
        # Training

        arch.train(dataset.train_x, dataset.train_y)


        ############################################
        # Results

        lresults.append((ahead, arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y)))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        if config is not None:
            updateprocess(config, ahead)

    arch.log_result(lresults)

    return lresults

def train_sequence2sequence_tf(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures with teacher forcing/attention

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """
    ahead = config['data']['ahead']

    if not type(ahead) == list:
        ahead = [1, ahead]

    dataset = Dataset(config=config['data'], data_path=wind_data_path)
    dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)
    # Reorganize data for teacher forcing
    dataset.teacher_forcing()

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] + 1
    else:
        odimensions = ahead

    lresults = []
    for iter in range(niter):

        config['idimensions'] = dataset.train_x.shape[1:]
        config['odimensions'] = odimensions
        arch = architecture(config, runconfig)

        if runconfig.multi == 1:
            arch.generate_model()
        else:
            with tf.device('/cpu:0'):
                arch.generate_model()

        if runconfig.verbose:
            arch.summary()
            arch.plot()
            dataset.summary()
            print()

        ############################################
        # Training
        arch.train([dataset.train_x, dataset.train_y_tf], dataset.train_y, [dataset.val_x, dataset.val_y_tf], dataset.val_y)

        ############################################
        # Results

        lresults.extend(arch.evaluate(dataset.val_x, dataset.val_y, dataset.test_x, dataset.test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save('-A%d-%d-R%02d' % (ahead[0], ahead[1], iter))

    arch.log_result(lresults)

    return lresults
