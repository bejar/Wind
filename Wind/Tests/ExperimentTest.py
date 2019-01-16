"""
.. module:: DataSetTest

DataSetTest
*************

:Description: DataSetTest

 Runs test for the configuration files in /TestConfigs
    

:Authors: bejar
    

:Version: 

:Created on: 26/11/2018 8:51 

"""

from Wind.Misc import load_config_file
from Wind.Train import TrainDispatch, RunConfig

__author__ = 'bejar'

def run_test(cfile):
    """
    Runs a configuration file
    """
    config = load_config_file(f"../TestConfigs/{cfile}.json", id=True)
    # print(config)
    dispatch = TrainDispatch()
    runconfig = RunConfig(impl=1,
                          verbose=True,
                          tboard=False,
                          best=True,
                          early=True,
                          multi=1,
                          proxy=False,
                          save=False,
                          remote=False,
                          info=True,
                          log=False
                          )
    train_process, architecture = dispatch.dispatch(config['arch']['mode'])

    lresults = train_process(architecture, config, runconfig)
    print(lresults)

if __name__ == '__main__':

    # run_test("config_CNN_s2s")
    # run_test("config_MLP_dir_reg")
    run_test("config_MLP_s2s")
    # run_test("config_MLP_s2s_rec")
    # run_test("config_persistence")
    # run_test("config_RNN_dir_reg")
    # run_test("config_RNN_ED_s2s")
    # run_test("config_RNN_s2s")
    # run_test("config_RNN_ED_s2s_att")
    # run_test("config_RNN_ED_s2s_dep")
    #run_test("config_SVM_dir_reg")
    # run_test("config_KNN_dir_reg")
    # run_test("config_KNN_dir_reg")

