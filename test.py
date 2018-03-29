# -*- coding: utf-8 -*-

import gedi.gpCalc as calc
import gedi.gpKernel as kernels
import numpy as np

def test_gedi():
    #data
    x = 10 * np.sort(np.random.rand(101))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))
    
    #lets define a kernel
    kernel0 = kernels.ExpSquared(10,1) * kernels.ExpSquared(5,0.3) + kernels.Exponential(1,1)
    
    #lets calculate the log-likelihood
    loglike = calc.likelihood(kernel0,x,y,yerr, kepler = False)

    return loglike