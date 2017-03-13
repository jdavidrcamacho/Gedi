# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:24:14 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl=Kernel
import Kernel_likelihood;reload(Kernel_likelihood);lk=Kernel_likelihood
import Kernel_optimization;reload(Kernel_optimization);opt=Kernel_optimization

import numpy as np
import matplotlib.pylab as pl

#####  DADOS INICIAS  #########################################################
np.random.seed(12345)
x = 10 * np.sort(np.random.rand(101))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

###############################################################################
#TESTS FOR THE LIKELIHOOD AND GRADIENT
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
print 'test 5 - optimization'
kernel1=kl.Exponential(10.0,1.0)+kl.WhiteNoise(1.0)
print 'kernel 1 ->', kernel1
likelihood1=lk.likelihood(kernel1,x,y,yerr)
print 'likelihood 1 ->', likelihood1

optimization1=opt.committed_optimization(kernel1,x,y,yerr,max_opt=2)
print 'kernel 1 final ->',optimization1[1]
print 'likelihood 1 final ->', optimization1[0]

###############################################################################
#TESTS FOR GRAPHICS

kernel2=kl.ExpSineSquared(10.0,1.0,10.0)+kl.Exponential(5.0,1.5)
xcalc=np.linspace(0,10,200)  

[mu,std]=lk.compute_kernel(kernel1,x,xcalc,y,yerr)
pl.figure()
pl.fill_between(xcalc, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(xcalc, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(xcalc, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(xcalc, mu, color="k", alpha=1, lw=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.title("Pre-optimization")
pl.xlabel("$x$")
pl.ylabel("$y$")

optimization1=opt.committed_optimization(kernel2,x,y,yerr,max_opt=10)
print 'final kernel ->',optimization1[1]
print 'final likelihood ->', optimization1[0]

[mu,std]=lk.compute_kernel(optimization1[1],x,xcalc,y,yerr)
pl.figure() 
pl.fill_between(xcalc, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(xcalc, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(xcalc, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(xcalc, mu, color="k", alpha=1, lw=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.title('Pos-optimization')
pl.xlabel("$x$")
pl.ylabel("$y$")