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
from Wind.DataBaseConfigurations import updateprocess
import numpy as np
from copy import deepcopy
from Wind.ErrorMeasure import ErrorMeasure

__author__ = 'bejar'


def train_dirregression(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    Multiorizon DIR strategy, an independent model for each horizon

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
            if 'descale' not in config['training'] or config['training']['descale']:
                lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y, scaler=dataset.scaler))
            else:
                lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            # Update result in db
            if config is not None and not runconfig.proxy:
                from Wind.DataBaseConfigurations import updateprocess
                updateprocess(config, ahead)

            arch.save(f'-A{ahead}-R{iter:02d}')
            del dataset

    arch.log_result(lresults)

    return lresults


def train_sjoint_sequence2sequence(architecture, config, runconfig):
    """
    Training process for architecture with multiple blocks of horizons

    Multihorizon SJOINT strategy

    The training is done separately for blocks of horizons (if block size is 1 this is dirregression)

    :return:
    """

    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    # Number of consecutive horizon elements to join in a prediction
    slice = config['data']['slice']

    # if (sahead - (iahead-1)) % slice != 0:
    #     raise NameError("SJOINT: slice has to be a divisor of the horizon length")


    lresults = []
    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    lmodels = []
    steps = [[i,j] for i,j in zip(range(iahead, sahead+1, slice), range(slice, sahead+slice+1,slice))]
    steps[-1][1] = sahead


    for iter in range(niter):
        # Loads the dataset once and slices the y matrix for training and evaluation
        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=[iahead, sahead], mode=architecture.data_mode, remote=runconfig.remote)

        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

        for recit, ahead in enumerate(steps):
            if runconfig.verbose:
                print('-----------------------------------------------------------------------------')
                print(f"Steps Ahead = {ahead}")

            ############################################
            # Model
            config['idimensions'] = train_x.shape[1:]
            config['odimensions'] = ahead[1] - ahead[0] + 1

            nconfig = deepcopy(config)
            nconfig['data']['ahead'] = ahead
            arch = architecture(nconfig, runconfig)

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
            # Training with the current slice
            arch.train(train_x, train_y[:,ahead[0]-1:ahead[1]], val_x, val_y[:,ahead[0]-1:ahead[1]])
            ############################################
            # Results
            if 'descale' not in config['training'] or config['training']['descale']:
                lresults.extend(arch.evaluate(val_x, val_y[:,ahead[0]-1:ahead[1]], test_x, test_y[:,ahead[0]-1:ahead[1]], scaler=dataset.scaler))
            else:
                lresults.extend(arch.evaluate(val_x, val_y[:,ahead[0]-1:ahead[1]], test_x, test_y[:,ahead[0]-1:ahead[1]]))

            print(strftime('%Y-%m-%d %H:%M:%S'))

            # Update result in db
            if config is not None and not runconfig.proxy:
                from Wind.DataBaseConfigurations import updateprocess
                updateprocess(config, ahead)

            arch.save(f"-{ahead[0]}-{ahead[1]}-S{recit:02d}-R{iter:02d}")

    arch.log_result(lresults)

    return lresults

def train_recursive_multi_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures with multi step recursive training

    Multihorizon SJOINT strategy with recursive prediccion, it only works with architectures prepared
    for recursive training

    Each block of horizons use the predicion of the previous block as part of the input

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
    steps = [[i,j] for i,j in zip(range(iahead, sahead+1, slice), range(slice, sahead+slice+1,slice))]
    steps[-1][1] = sahead

    ### Accumulated recursive predictions for train, validation and test
    rec_train_pred_x = None
    rec_val_pred_x = None
    rec_test_pred_x = None

    for iter in range(niter):
        dataset = Dataset(config=config['data'], data_path=wind_data_path)
        dataset.generate_dataset(ahead=[iahead, sahead], mode=architecture.data_mode, remote=runconfig.remote)
        dataset.summary()
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_data_matrices()

        for recit, ahead in enumerate(steps):
            if runconfig.verbose:
                print('-----------------------------------------------------------------------------')
                print(f"Steps Ahead = {ahead}")

            # Dataset (the y matrices depend on the slice used for prediction


            ############################################
            # Model

            config['idimensions'] = train_x.shape[1:]
            config['odimensions'] = ahead[1] - ahead[0] + 1
            # Dimensions for the recursive input

            # For evaluating we need to pass the range of columns of the current iteration
            config['rdimensions'] = recit * slice

            # print(f"IDim:{config['idimensions']} ODim:{config['odimensions']} RDim:{config['rdimensions']}")

            nconfig = deepcopy(config)
            nconfig['data']['ahead'] = ahead
            arch = architecture(nconfig, runconfig)

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
            # print(f'Ahead: {ahead[0] - 1}  {ahead[1]}')
            if config['rdimensions'] == 0:
                arch.train(train_x, train_y[:,ahead[0]-1:ahead[1]], val_x, val_y[:,ahead[0]-1:ahead[1]])
            else:
                # Train using the predictions of the previous iteration
                arch.train([train_x, rec_train_pred_x], train_y[:,ahead[0]-1:ahead[1]],
                           [val_x, rec_val_pred_x], val_y[:,ahead[0]-1:ahead[1]])

            ############################################
            # Results and Add the new predictions to the saved ones
            if config['rdimensions'] == 0:
                if 'descale' not in config['training'] or config['training']['descale']:
                    lresults.extend(arch.evaluate(val_x, val_y[:,ahead[0]-1:ahead[1]], test_x, test_y[:,ahead[0]-1:ahead[1]], scaler=dataset.scaler))
                else:
                    lresults.extend(arch.evaluate(val_x, val_y[:,ahead[0]-1:ahead[1]], test_x, test_y[:,ahead[0]-1:ahead[1]]))
                rec_train_pred_x = arch.predict(train_x)
                rec_val_pred_x = arch.predict(val_x)
                rec_test_pred_x = arch.predict(test_x)
                # print(f"TRshape:{train_x.shape} Vshape:{val_x.shape} TSshape:{test_x.shape}")
                # print(f"RTR:{rec_train_pred_x.shape} RV:{rec_val_pred_x.shape} RTS:{rec_test_pred_x.shape}")
            else:

                if 'descale' not in config['training'] or config['training']['descale']:
                    lresults.extend(arch.evaluate([val_x, rec_val_pred_x], val_y[:,ahead[0]-1:ahead[1]],
                                                  [test_x, rec_test_pred_x], test_y[:,ahead[0]-1:ahead[1]], scaler=dataset.scaler))
                else:
                    lresults.extend(arch.evaluate([val_x, rec_val_pred_x], val_y[:,ahead[0]-1:ahead[1]],
                                                  [test_x, rec_test_pred_x], test_y[:,ahead[0]-1:ahead[1]]))


                rec_train_pred_x = np.concatenate((rec_train_pred_x, arch.predict([train_x, rec_train_pred_x])), axis=1)
                rec_val_pred_x = np.concatenate((rec_val_pred_x, arch.predict([val_x, rec_val_pred_x])), axis=1)
                rec_test_pred_x = np.concatenate((rec_test_pred_x, arch.predict([test_x, rec_test_pred_x])), axis=1)
                # print(f"TRshape:{train_x.shape} Vshape:{val_x.shape} TSshape:{test_x.shape}")
                # print(f"RTR:{rec_train_pred_x.shape} RV:{rec_val_pred_x.shape} RTS:{rec_test_pred_x.shape}")

            print(strftime('%Y-%m-%d %H:%M:%S'))

            arch.save(f"-{ahead[0]}-{ahead[1]}-S{recit:02d}-R{iter:02d}")

    arch.log_result(lresults)

    return lresults


def train_gradient_boosting_sequence2sequence(architecture, config, runconfig):
    """
    Training process for sequence 2 sequence architectures

    Mutihorizon MIMO/DIRJOINT strategy plus gradient boosting

    Generates a sequence of models that train over the difference of the previous prediction

    :param architecture:
    :param config:
    :param runconfig:
    :return:
    """

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

        # Training using gradient boosting (kindof)
        # Generate a bunch of models
        boost_train_pred = []
        boost_val_pred = []
        boost_test_pred = []
        n_train_y = train_y
        n_val_y = val_y
        n_test_y = test_y
        alpha = config['arch']['alpha']
        decay = config['arch']['decay']
        for nm in range(config['arch']['nmodels']):
            arch.train(train_x, n_train_y, val_x, n_val_y)

            # Prediction of the current model
            boost_train_pred.append(arch.predict(train_x))
            boost_val_pred.append(arch.predict(val_x))
            boost_test_pred.append(arch.predict(test_x))

            # Compute the prediction of the combination of models
            # Prediction of the first model
            boost_train_predict_y = boost_train_pred[0]
            boost_val_predict_y = boost_val_pred[0]
            boost_test_predict_y = boost_test_pred[0]
            for m in range(1,len(boost_train_pred)):
                boost_train_predict_y += (alpha * boost_train_pred[m])
                boost_val_predict_y += (alpha * boost_val_pred[m])
                boost_test_predict_y += (alpha * boost_test_pred[m])

            # Residual of the prediction for the next step
            n_train_y = train_y - boost_train_predict_y
            n_val_y = val_y - boost_val_predict_y
            # print(ErrorMeasure().compute_errors(val_y[:, 0], boost_val_predict_y[:, 0], test_y[:, 0], boost_test_predict_y[:, 0]))
            alpha *= decay

            # Reset the model
            arch = architecture(config, runconfig)
            if runconfig.multi == 1:
                arch.generate_model()
            else:
                with tf.device('/cpu:0'):
                    arch.generate_model()
            # For now the model is not saved
            arch.save(f'-{ahead[0]}-{ahead[1]}-R{nm}')

        ############################################
        # Results

        # Maintained to be compatible with old configuration files
        if type(config['data']['ahead'])==list:
            iahead = config['data']['ahead'][0]
            ahead = (config['data']['ahead'][1] - config['data']['ahead'][0]) + 1
        else:
            iahead = 1
            ahead = config['data']['ahead']

        itresults = []

        for i, p in zip(range(1, ahead + 1), range(iahead, config['data']['ahead'][1]+1)):

            if 'descale' not in config['training'] or config['training']['descale']:
                itresults.append([p]  + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                                   boost_val_predict_y[:, i - 1],
                                                                   test_y[:, i - 1],
                                                                   boost_test_predict_y[:, i - 1], scaler=dataset.scaler))
            else:
                itresults.append([p]  + ErrorMeasure().compute_errors(val_y[:, i - 1],
                                                                   boost_val_predict_y[:, i - 1],
                                                                   test_y[:, i - 1],
                                                                   boost_test_predict_y[:, i - 1]))

        lresults.extend(itresults)

        print(strftime('%Y-%m-%d %H:%M:%S'))


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

    ahead = config['data']['ahead'] if (type(config['data']['ahead']) == list) else [1, config['data']['ahead']]
    #if 'aggregate' in config['data']:
    #    step = config['data']['aggregate']['step']
    #    ahead = [ahead[0], ahead[1]//step]

    if 'iter' in config['training']:
        niter = config['training']['iter']
    else:
        niter = 1

    if type(ahead) == list:
        odimensions = ahead[1] - ahead[0] + 1
        if 'aggregate' in config['data'] and 'y' in config['data']['aggregate']:
            step = config['data']['aggregate']['y']['step']
            odimensions //= step
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

        if 'descale' not in config['training'] or config['training']['descale']:
            lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y,scaler=dataset.scaler))
        else:
            lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save(f'-{ahead[0]}-{ahead[1]}-R{iter}')

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

        if 'descale' not in config['training'] or config['training']['descale']:
            lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y,scaler=dataset.scaler))
        else:
            lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y))

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
        if 'descale' not in config['training'] or config['training']['descale']:
            lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y,scaler=dataset.scaler))
        else:
            lresults.append([ahead] + arch.evaluate(val_x, val_y, test_x, test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        if config is not None:
            updateprocess(config, ahead)

    arch.log_result(lresults)

    return lresults


def train_sckit_sequence2sequence(architecture, config, runconfig):
    """
    Training process for architecture with direct regression of ahead time steps

    :return:
    """

    ahead = config['data']['ahead'] if (type(config['data']['ahead']) == list) else [1, config['data']['ahead']]
    if type(config['data']['ahead']) == list:
        iahead, sahead = config['data']['ahead']
    else:
        iahead, sahead = 1, config['data']['ahead']

    # lresults = []


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
    if 'descale' not  in config['training'] or config['training']['descale']:
        lresults = arch.evaluate(val_x, val_y, test_x, test_y, scaler=dataset.scaler)
    else:
        lresults = arch.evaluate(val_x, val_y, test_x, test_y)

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

        if 'descale' not in config['training'] or config['training']['descale']:
            lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y,scaler=dataset.scaler))
        else:
            lresults.extend(arch.evaluate(val_x, val_y, test_x, test_y))

        print(strftime('%Y-%m-%d %H:%M:%S'))

        arch.save('-A%d-%d-R%02d' % (ahead[0], ahead[1], iter))

    arch.log_result(lresults)

    return lresults
