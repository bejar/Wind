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

from Wind.Util import load_config_file
from Wind.Train import TrainDispatch, RunConfig
__author__ = 'bejar'

def run_test(cfile):
    """
    Runs a configuration file
    """
    config = load_config_file(f"../TestConfigs/{cfile}.json", id=True)
    print(config)
    dispatch = TrainDispatch()
    runconfig = RunConfig(1,True, False,True,True,1,False,False,False)
    train_process, architecture = dispatch.dispatch(config['arch']['mode'])

    lresults = train_process(architecture, config, runconfig)
    print(lresults)

if __name__ == '__main__':

    # run_test("config_CNN_s2s")
    # run_test("config_MLP_dir_reg")
    # run_test("config_MLP_s2s")
    # run_test("config_persistence")
    #run_test("config_RNN_dir_reg")
    # run_test("config_RNN_ED_s2s")
    run_test("config_RNN_s2s")