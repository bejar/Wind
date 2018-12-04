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
import numpy as np

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

            # Dataset
            dataset = Dataset(config=config['data'], data_path=wind_data_path)
            dataset.generate_dataset(ahead=[ahead, ahead], mode=architecture.data_mode, remote=runconfig.remote)
            train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

            ############################################
            # Model
            config['idimensions'] = train_x.shape[1:]

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
            arch.train(train_x, train_y, val_x, val_y)

            ############################################
            # Results

            r2val, r2test = arch.evaluate(val_x, val_y, test_x, test_y)
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


def train_recursive_multi_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures with multi step recursive training

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """
    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    slice = config['data']['slice']

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    lresults = []
    lmodels = []
    steps = [[i,j] for i,j in zip(range(iahead, iahead+1, slice), range(slice, iahead+slice+1,slice))]
    steps[-1][1] = sahead

    ### Accumulated recursive predictions for train, validation and test
    rec_train_pred_x = None
    rec_val_pred_x = None
    rec_test_pred_x = None

    for iter in range(niter):
        for ahead in steps:

            if runconfig.verbose:
                print('-----------------------------------------------------------------------------')
                print(f"Steps Ahead = {ahead}")

            # Dataset
            dataset = Dataset(config=config['data'], data_path=wind_data_path)
            dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)
            train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

            ############################################
            # Model

            config['idimensions'] = train_x.shape[1:]
            config['odimensions'] = ahead[1] - ahead[0]
            config['rdimensions'] = ahead[0] - 1

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
            if config['rdimensions'] == 0:
                arch.train(train_x, train_y, val_x, val_y)
            else:
                # Train using the predictions of the previous iteration
                arch.train([train_x, rec_train_pred_x], train_y, [val_x, rec_val_pred_x], val_y)

            ############################################
            # Results and Add the new predictions to the saved ones
            if config['rdimensions'] == 0:
                lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y))
                rec_train_pred_x = arch.predict(train_x)
                rec_val_pred_x = arch.predict(val_x)
                rec_test_pred_x = arch.predict(test_x)

            else:
                lresults.extend(arch.evaluate([val_x, rec_val_pred_x], val_y,
                                              [test_x, rec_test_pred_x], dataset.test_y))
                rec_train_pred_x = np.concatenate((rec_train_pred_x, arch.predict(train_x)))
                rec_val_pred_x = np.concatenate((rec_val_pred_x, arch.predict(val_x)))
                rec_test_pred_x = np.concatenate((rec_test_pred_x, arch.predict(test_x)))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            arch.save(f"-{ahead[0]}-{ahead[1]}-R{iter:02d}")

    arch.log_result(lresults)

    return lresults


def train_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures

    Mutihorizon MIMO/DIRJOINT strategy

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """

    # if type(config['data']['ahead']) == list:
    #     iahead, sahead = config['data']['ahead']
    # else:
    #     iahead, sahead = 1, config['data']['ahead']

    ahead = config['data']['ahead'] if (type(config['data']['ahead']) == list) else [1, config['data']['ahead']]

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] + 1
    else:
        odimensions = ahead

    # Dataset
    dataset = Dataset(config=config['data'], data_path=wind_data_path)
    dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)
    train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

    if type(train_x) != list:
        config['idimensions'] = train_x.shape[1:]
    else:
        config['idimensions'] = [d.shape[1:] for d in train_x]
    config['odimensions'] = odimensions

    lresults = []
    for iter in range(niter):

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
        arch.train(train_x, train_y, val_x, val_y)

        ############################################
        # Results

        lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y))

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

        # Dataset
        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=[ahead, ahead], mode=architecture.data_mode, remote=runconfig.remote)
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

        # Architecture
        arch = architecture(config, runconfig)

        if runconfig.verbose:
            dataset.summary()


        val_r2, test_r2 = arch.evaluate(val_x, val_y, test_x, test_y)
        lresults.append((ahead, val_r2, test_r2))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        # Update result in db
        if config is not None:
            updateprocess(config, ahead)

        del dataset

    arch.log_result(lresults)
    return lresults


def train_sckit_dirregression(architecture, config, runconfig):
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
            print('************************************************************')
            print(f'Steps Ahead = {ahead} ')

        # Dataset
        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

        ############################################
        # Model

        arch = architecture(config, runconfig)
        arch.generate_model()

        if runconfig.verbose:
            arch.summary()
            dataset.summary()
            print()

        ############################################
        # Training
        arch.train(train_x, train_y, val_x, val_y)

        ############################################
        # Results
        val_r2, test_r2 = arch.evaluate(val_x, val_y, test_x, test_y)
        lresults.append((ahead, val_r2, test_r2))

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
    if not dataset.is_teacher_force():
        raise NameError("S2S teacher force: invalid data matrix")

    dataset.generate_dataset(ahead=ahead, mode=architecture.data_mode, remote=runconfig.remote)

    # Reorganize data for teacher forcing
    # dataset.teacher_forcing()

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
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()
        if type(train_x) != list:
            config['idimensions'] = train_x.shape[1:]
        else:
            config['idimensions'] = [d.shape[1:] for d in train_x]

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


        arch.train(train_x, train_y, val_x, val_y)

        ############################################
        # Results

        lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save('-A%d-%d-R%02d' % (ahead[0], ahead[1], iter))

    arch.log_result(lresults)

    return lresults
