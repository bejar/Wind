"""
.. module:: SCKArchitecture

SCKArchitecture
*************

:Description: SCKArchitecture

 Metaclass for scikit learn classifiers using direct regression

:Authors: bejar
    

:Version: 

:Created on: 04/12/2018 7:46 

"""

from Wind2.Architectures.Architecture import Architecture
from Wind2.ErrorMeasure import ErrorMeasure
import h5py

__author__ = 'bejar'


class SCKArchitecture(Architecture):
    """
    Class for all the scikit models using direct regression

    """
    ## data mode 2 dimensional input and only one output
    data_mode = ('2D', '0D') #'svm'
    modname = 'SCKDIRREG'

    def train(self, train_x, train_y, val_x, val_y):
        """
        Trains the model

        :return:
        """
        self.model.fit(train_x, train_y)

    def evaluate(self, val_x, val_y, test_x, test_y, scaler=None, save_errors=None):
        """
        Evaluates the training
        :param save_errors:
        :return:
        """
        val_yp = self.model.predict(val_x)
        test_yp = self.model.predict(test_x)

        if save_errors is not None:
            f = h5py.File(f'errors{self.modname}-S{self.config["data"]["datanames"][0]}{save_errors}.hdf5', 'w')
            dgroup = f.create_group('errors')
            dgroup.create_dataset('val_y', val_y.shape, dtype='f', data=val_y, compression='gzip')
            dgroup.create_dataset('val_yp', val_yp.shape, dtype='f', data=val_yp, compression='gzip')
            dgroup.create_dataset('test_y', test_y.shape, dtype='f', data=test_y, compression='gzip')
            dgroup.create_dataset('test_yp', test_yp.shape, dtype='f', data=test_y, compression='gzip')
            if scaler is not None:
                # Unidimensional vectors
                dgroup.create_dataset('val_yu', val_y.shape, dtype='f', data=scaler.inverse_transform(val_y.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('val_ypu', val_yp.shape, dtype='f', data=scaler.inverse_transform(val_yp.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('test_yu', test_y.shape, dtype='f', data=scaler.inverse_transform(test_y.reshape(-1, 1)), compression='gzip')
                dgroup.create_dataset('test_ypu', test_yp.shape, dtype='f', data=scaler.inverse_transform(test_yp.reshape(-1, 1)), compression='gzip')

        return ErrorMeasure().compute_errors(val_y, val_yp, test_y, test_yp)


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
        

