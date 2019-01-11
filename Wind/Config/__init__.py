"""
.. module:: __init__.py

__init__.py
*************

:Description: __init__.py

    

:Authors: bejar
    

:Version: 

:Created on: 15/06/2017 11:15 

"""

__author__ = 'bejar'

from .Paths import wind_path, wind_data_path, wind_data_ext, wind_models_path, bsc_path, wind_jobs_path, wind_res_path, \
    jobs_root_path, jobs_code_path


__all__ = ['wind_path', 'wind_data_path', 'wind_data_ext', 'wind_models_path', 'bsc_path', 'jobs_root_path', 'jobs_code_path']
