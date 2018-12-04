"""
.. module:: SVMDirRegressionArchitecture

SVMDirRegressionArchitecture
*************

:Description: SVMDirRegressionArchitecture

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:33 

"""

from Wind.Architectures.SCKArchitecture import SCKArchitecture
from sklearn.svm import SVR

__author__ = 'bejar'


class SVMDirRegressionArchitecture(SCKArchitecture):
    data_mode = ('2D', '0D') #'svm'
    modname = 'SVMDIRREG'

    def generate_model(self):
        """
        Generates the model

        :return:
        """
        kernel = self.config['arch']['kernel']
        C = self.config['arch']['C']
        epsilon = self.config['arch']['epsilon']
        degree = self.config['arch']['degree']
        coef0 = self.config['arch']['coef0']

        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, coef0=coef0)

    # def summary(self):
    #     """
    #     Model summary
    #     :return:
    #     """
    #     kernel = self.config['arch']['kernel']
    #     C = self.config['arch']['C']
    #     epsilon = self.config['arch']['epsilon']
    #     degree = self.config['arch']['degree']
    #     coef0 = self.config['arch']['coef0']
    #
    #     print(
    #     f"lag: {self.config['data']['lag']} /kernel: {kernel} /C: {C} /epsilon: {epsilon} /degree: {degree} /coef0 {coef0}")

    def log_result(self, result):
        """
        logs a result from the model

        :param result:
        :return:
        """
        for i, r2val, r2test in result:
            print(f"{self.config['arch']['mode']} | AH={i} KRNL={self.config['arch']['kernel']}  C={self.config['arch']['C']:3.5f} "
                  f"EPS={self.config['arch']['epsilon']:3.5f} DEG={self.config['arch']['degree']} "
                  f"COEF0={self.config['arch']['coef0']} "
                  f"R2V = {r2val:3.5f}, R2T = {r2test:3.5f}"
                  )
            # print('%s |  AH=%d, KRNL= %s, C= %3.5f, EPS= %3.5f, DEG=%d, COEF0= %d, R2V = %3.5f, R2T = %3.5f' %
            #       (self.config['arch']['mode'], i,
            #        self.config['arch']['kernel'],
            #        self.config['arch']['C'],
            #        self.config['arch']['epsilon'],
            #        self.config['arch']['degree'],
            #        self.config['arch']['coef0'],
            #        r2val, r2test
            #        ))

