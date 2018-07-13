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

class Architecture:
    config = None
    runconfig = None
    model = None

    def __init__(self, config, runconfig):
        """
        Stores the configuration for the model
        :param config:
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

        :return:
        """
        raise NameError('ERROR: Not implemented')

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        raise NameError('ERROR: Not implemented')


    def summary(self):
        """
        Model summary
        :return:
        """
        raise NameError('ERROR: Not implemented')

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the training
        :return:
        """
        raise NameError('ERROR: Not implemented')


    def log_result(self, result):
        """
        logs a result from the model

        :param result:
        :return:
        """
        raise NameError('Error: Not implemented')

    def save(self, postfix):
        """
        Saves model to a file
        :return:
        """
        raise NameError('Error: Not implemented')
