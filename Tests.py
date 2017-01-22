# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:24:14 2016

@author: camacho
"""
import Kernel_likelihood as kl
import Kernel_optimization as opt

import numpy as np
#####  DADOS INICIAS  #########################################################
np.random.seed(1234)
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
###############################################################################

print 'test 1'
kernel1=kl.Kernel.ExpSquared(11.0,7.0)
kl.likelihood(kernel1,x,y,yerr)
print kl.gradient_likelihood(kernel1,x,y,yerr)

print 'test 2'
kernel2=kl.Kernel.ExpSineSquared(10.1,1.2,5.1)
kl.likelihood(kernel2,x,y,yerr)
print kl.gradient_likelihood(kernel2,x,y,yerr)

print 'test 3'
kernel3=kl.Kernel.ExpSquared(10.2,7.1)+kl.Kernel.ExpSineSquared(10.1,1.2,5.1)
kl.likelihood(kernel3,x,y,yerr)
print kl.gradient_likelihood(kernel3,x,y,yerr)

