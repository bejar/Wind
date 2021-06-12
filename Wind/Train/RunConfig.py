"""
.. module:: RunConfig

RunConfig
*************

:Description: RunConfig

    Object to store information from the flags passed to the script for training

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:09 

"""

__author__ = 'bejar'


class RunConfig:
    """Class Runconfig

    Stores information from the flags of the script and are not in the configuration file of the experiment

    """
    ## Implementation to use for RNN
    impl = 1
    ## Activates TF verbose output and other information
    verbose = False
    ## Generate output for tensorboard
    tboard = False
    ## Keep the model with best validation accuracy
    best = True
    ## Early stopping
    early = True
    ## Multi GPU training
    multi = False
    ## Get experiment configuration using the proxy
    proxy = False
    ## Save the final model
    save = False
    ## Get the data from the remote server
    remote = False
    ## Print info of dataset and model at the end of training
    info = False
    ## Not yet used
    log = None

    def __init__(self, impl=1, verbose=False, tboard=False, best=True, early=True, multi=False, proxy=False, save=False,
                 remote=False, info=False, log=False):
        """ Constructor

        Stores the parameters in the object attributes

        :param impl:
        :param verbose:
        :param tboard:
        :param best:
        :param early:
        :param multi:
        :param proxy:
        :param save:
        :param remote:
        :param info:
        :param log:
        """
        self.impl = impl
        self.verbose = verbose
        self.tboard = tboard
        self.best = best
        self.early = early
        self.multi = multi
        self.proxy = proxy
        self.save = save
        self.remote = remote
        self.info = info
        self.log = log
