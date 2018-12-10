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
    modname = None
    data_mode = None

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
        print("--------- Architecture parameters -------")
        print(f"{self.modname}")
        for c in self.config['arch']:
            print(f"# {c} = {self.config['arch'][c]}")
        if 'training' in self.config:
            print("--------- Training parameters -------")
            for c in self.config['training']:
                print(f"# {c} = {self.config['training'][c]}")
            print("---------------------------------------")

    def evaluate(self, val_x, val_y, test_x, test_y):
        """
        Evaluates the trained model for validation and test
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

        for i, r2val, r2test in result:
            print(f"{self.config['arch']['mode']} | AH={i} R2V = {r2val:3.5f} R2T = {r2test:3.5f}"
                  )

    def save(self, postfix):
        """
        Saves model to a file
        :return:
        """
        pass

    def plot(self):
        """
        Plots the model architecture
        :return:
        """
        pass
