"""
.. module:: Architecture

Architecture
*************

:Description: Architecture

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:23 

"""

__author__ = 'bejar'

from Wind.ErrorMeasure import ErrorMeasure

class Architecture:
    """Architecture

    Class for all the architectures
    """
    ## Name to use for generating files from the architecture and experiment
    modfile = None
    ## Stores the experiment configuration
    config = None
    ## Stores the configuration flags passed to the training script
    runconfig = None
    ## Stores the architecture generated
    model = None
    ## Stores the name of the architecture
    modname = None
    ## Stores a pair that codes how the input and output data matrices have to be generated
    data_mode = None

    def __init__(self, config, runconfig):
        """Constructor
        Stores the configuration for the model

        :param config:
        :param runconfig:
        """
        self.config = config
        self.runconfig = runconfig

    def add_config(self, key, value):
        """
        Adds additional configuration that could be needed by the model

        :param key:
        :param value:
        :return:
        """
        self.config[key] = value

    def generate_model(self):
        """
        Generates the model

        Abstract
        :return:
        """
        raise NameError('ERROR: Not implemented')

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        Abstract

        :param train_x:
        :param train_y:
        :param val_x:
        :param val_y:
        :return:
        """
        raise NameError('ERROR: Not implemented')

    def summary(self):
        """Model summary

        prints all the fields stored in the configuration for the experiment

        :return:
        """
        print("--------- Architecture parameters -------")
        print(f"{self.modname}")
        for c in self.config['arch']:
            print(f"# {c} = {self.config['arch'][c]}")
        print("--------- Data parameters -------")
        for c in self.config['data']:
            print(f"# {c} = {self.config['data'][c]}")
        if 'training' in self.config:
            print("--------- Training parameters -------")
            for c in self.config['training']:
                print(f"# {c} = {self.config['training'][c]}")
            print("---------------------------------------")
        self.model.summary()

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None):
        """
        Evaluates the trained model for validation and test

        Abstract
        :param val_x:
        :param val_y:
        :param test_x:
        :param test_y:
        :return:
        """
        raise NameError('ERROR: Not implemented')

    def log_result(self, result):
        """
        logs a result from the model (basic results)

        :param result:
        :return:
        """
        if self.runconfig.info:
            self.summary()

        if not 'iter' in self.config['training']:
            nres = len(result)
        else:
            nres = len(result)//self.config['training']['iter']

        ErrorMeasure().print_errors(self.config['arch']['mode'], nres, result)


    def save(self, postfix):
        """
        Saves model to a file

        Abstract

        :param postfix:
        :return:
        """
        pass

    def plot(self):
        """
        Plots the model architecture

        Abstract
        :return:
        """
        pass
