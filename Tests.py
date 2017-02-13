# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:24:14 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl=Kernel
import Kernel_likelihood;reload(Kernel_likelihood);lk=Kernel_likelihood
import Kernel_optimization;reload(Kernel_optimization);opt=Kernel_optimization

import george
import george.kernels as ge

import numpy as np

#####  DADOS INICIAS  #########################################################
np.random.seed(12345)
x = 10 * np.sort(np.random.rand(101))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
###############################################################################
#TESTS FOR THE LIKELIHOOD AND GRADIENT

print '########## Just tests ##############'

print 'test 1'
kernel1=kl.ExpSquared(11.0,7.0)
kernel1_test1= lk.likelihood(kernel1,x,y,yerr)
kernel1_test2= lk.gradient_likelihood(kernel1,x,y,yerr)
print 'kernel =',kernel1
print 'likelihood =',kernel1_test1
print 'gradients =',kernel1_test2

print 'test 2'
kernel2=kl.ExpSineSquared(10.1,1.2,5.1)
kernel2_test1= lk.likelihood(kernel2,x,y,yerr)
kernel2_test2= lk.gradient_likelihood(kernel2,x,y,yerr)
print 'kernel =',kernel2
print 'likelihood =',kernel2_test1
print 'gradients =',kernel2_test2

print 'test 3'
kernel3=kl.ExpSquared(10.2,7.1)+kl.ExpSineSquared(10.1,1.2,5.1)
kernel3_test1= lk.likelihood(kernel3,x,y,yerr)
kernel3_test2= lk.gradient_likelihood(kernel3,x,y,yerr)
print 'kernel =',kernel3
print 'likelihood =',kernel3_test1
print 'gradients =',kernel3_test2

print 'test 4'
kernel4=kl.ExpSquared(10.2,7.1)*kl.ExpSineSquared(10.1,1.2,5.1)
kernel4_test1= lk.likelihood(kernel4,x,y,yerr)
kernel4_test2= lk.gradient_likelihood(kernel4,x,y,yerr)
print 'kernel =',kernel4
print 'likelihood =',kernel4_test1
print 'gradients =',kernel4_test2

print '#####################################'

###############################################################################
#TESTS FOR THE OPTIMIZATION

#print '########## Calculations from george ##########'
##kernel = ge.ExpSine2Kernel(2.0/1.1**2, 7.1)
#kernel = ge.ExpSine2Kernel(2.0/1.7**2,5.0)
#gp = george.GP(kernel)
#gp.compute(x,yerr)
#
#print 'initial kernel_george ->', kernel
#print 'initial likelihood_george ->', gp.lnlikelihood(y)
#### OPTIMIZE HYPERPARAMETERS
#import scipy.optimize as op
## Define the objective function (negative log-likelihood in this case).
#def nll(p):
#    # Update the kernel parameters and compute the likelihood.
#    gp.kernel[:] = p
#    ll = gp.lnlikelihood(y, quiet=True)
#
#    # The scipy optimizer doesn't play well with infinities.
#    return -ll if np.isfinite(ll) else 1e25
#
## And the gradient of the objective function.
#def grad_nll(p):
#    # Update the kernel parameters and compute the likelihood.
#    gp.kernel[:] = p
#    return -gp.grad_lnlikelihood(y, quiet=True)
#
## You need to compute the GP once before starting the optimization.
#gp.compute(x,yerr)
#
#p0 = gp.kernel.vector
#results = op.minimize(nll, p0, jac=grad_nll)
#
## Update the kernel and print the final log-likelihood.
#gp.kernel[:] = results.x
#
#print 'final kernel_george ->', kernel #kernel final
#print 'final likelihood_george ->', gp.lnlikelihood(y)
#print '#####################################'
#
#
#print '########## Calculations from gedi ##########'
#kernel1=kl.ExpSineGeorge(2.0/1.7**2,5.0)
#print 'initial kernel',kernel2
#kernel1_result1=lk.likelihood(kernel1,x,y,yerr)
#
##optimization
#print 'EXAMPLE 1a - BFGS'
#kernel1_resulta=opt.optimization(kernel1,x,y,yerr,method='BFGS')
#print(kernel1_resulta)
#
##optimization
#print 'EXAMPLE 1b - SDA'
#kernel1_resultb=opt.optimization(kernel1,x,y,yerr,method='SDA')
#print(kernel1_resultb)
#
##optimization
#print 'EXAMPLE 1c - RPROP'
#kernel1_resultc=opt.optimization(kernel1,x,y,yerr,method='RPROP')
#print(kernel1_resultc)
#
##optimization
#print 'EXAMPLE 1d - altSDA'
#kernel1_resultc=opt.optimization(kernel1,x,y,yerr,method='altSDA')
#print(kernel1_resultc)
#print '#####################################'
#
#
#print '########## Calculations from gedi ##########'
#kernel2=kl.ExpSineSquared(2.1,1.5,10.2)
#print 'initial kernel',kernel2
#kernel2_result1=lk.likelihood(kernel2,x,y,yerr)
#
##optimization
#print 'EXAMPLE 2a - BFGS'
#kernel2_resulta=opt.optimization(kernel2,x,y,yerr,method='BFGS')
#print kernel2_resulta
#
##optimization
#print 'EXAMPLE 2b - SDA'
#kernel2_resultb=opt.optimization(kernel2,x,y,yerr,method='SDA')
#print kernel2_resultb
#
##optimization
#print 'EXAMPLE 2c - RPROP'
#kernel2_resultc=opt.optimization(kernel2,x,y,yerr,method='RPROP')
#print(kernel2_resultc)
#
##optimization
#print 'EXAMPLE 2d - altSDA'
#kernel2_resultc=opt.optimization(kernel2,x,y,yerr,method='altSDA')
#print(kernel2_resultc)
#print '#####################################'
#
#
#print '########## Calculations from gedi ##########'
#kernel3=kl.ExpSineSquared(1.5,0.6,11.0)
#print 'initial kernel',kernel3
#kernel3_result1=lk.likelihood(kernel3,x,y,yerr)
#
##optimization
#print 'EXAMPLE 3a - BFGS'
#kernel3_resulta=opt.optimization(kernel3,x,y,yerr,method='BFGS')
#print kernel3_resulta
#
##optimization
#print 'EXAMPLE 3b - SDA'
#kernel3_resultb=opt.optimization(kernel3,x,y,yerr,method='SDA')
#print kernel3_resultb
#
##optimization
#print 'EXAMPLE 3c - RPROP'
#kernel3_resultc=opt.optimization(kernel3,x,y,yerr,method='RPROP')
#print(kernel3_resultc)
#
##optimization
#print 'EXAMPLE 3c - altSDA'
#kernel3_resultc=opt.optimization(kernel3,x,y,yerr,method='altSDA')
#print(kernel3_resultc)
#print '#####################################'
#
#### Conclusion: I don't trust RPROP's logic
#### I need to check were the None comes from, I have no idea for now...
