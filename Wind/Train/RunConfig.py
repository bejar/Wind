"""
.. module:: RunConfig

RunConfig
*************

:Description: RunConfig

    

:Authors: bejar
    

:Version: 

:Created on: 06/07/2018 8:09 

"""

__author__ = 'bejar'


class RunConfig:
    impl = 1
    verbose = False
    tboard = False
    best = True
    early = True
    multi = False
    proxy = False
    save = False
    remote = False
    info = False
    log = None

    def __init__(self, impl, verbose, tboard, best, early, multi, proxy, save, remote, info, log):
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
