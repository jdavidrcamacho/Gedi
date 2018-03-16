# -*- coding: utf-8 -*-

##### Necessary scripts to everything work #####
"""
    Package to analyze radial velocity measurements using Gaussian processes.
    
    gpObject: Initial object to work with Gaussian processes,
	contains the log likelihood calculation.
    kernels: contains all the developed kernels.

"""

from gedi import gpCalc
from gedi import gpKernel
