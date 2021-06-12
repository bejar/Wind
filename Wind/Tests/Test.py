"""
.. module:: Test

Test
*************

:Description: Test

    

:Authors: bejar
    

:Version: 

:Created on: 09/11/2018 11:52 

"""

from Wind.Architectures.RNNS2SArchitecture import RNNS2SArchitecture
from Wind.Misc import load_config_file
from Wind.Train import RunConfig

__author__ = 'bejar'

if __name__ == '__main__':
    config = load_config_file("../examples/config_RNN_ED_s2s.json", id=True)
    config['idimensions'] = (12, 6)
    config['odimensions'] = 12
    runconfig = RunConfig(1, 0, False, False, False, False, False, False, False)

    print(config)
    net = RNNS2SArchitecture(config, runconfig)
    net.generate_model()
    net.summary()
