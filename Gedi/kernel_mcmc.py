# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:51:50 2017

@author: camacho
"""

import kernel as kl
import kernel_likelihood as lk

import numpy as np
import inspect

##### markov chain monte carlo #####
"""
    MCMC() perform the markov chain monte carlo to find the optimal parameters
of a given kernel.
    The algorithm needs improvements as it is very inefficient.

    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments
parameters = the interval of the kernel parameters (check the Tests.py
            understand it better)
runs = the number of times the mcmc runs, 50000 by definition, its a lot but
    this version of the mcmc its still very inefficient, I hope to release a
    better one in the future
        
""" 
def MCMC(kernel,x,y,yerr,parameters,runs=50000):
    #to not loose que original kernel and data
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr

    initial_params= [0]*len(parameters)
    for i in range(len(parameters)):
        initial_params[i]=np.random.uniform(parameters[i][0],parameters[i][1])    
    first_kernel=new_kernel(kernelFIRST,initial_params)        
    first_likelihood=lk.likelihood(first_kernel,x,y,yerr)  
    print first_kernel,first_likelihood        
    
    i=0
    step=5e-3 #a better way to define the step is needed
    running_logs=[]
    params_number=len(parameters)
    params_list = [[] for _ in range(params_number)]
    
    accepts=0;rejects=0
    while i<runs:
        u=np.random.uniform(0,1)
        guess_params=[np.abs(n+(step)*np.random.randn()) for n in initial_params]    
        second_kernel=new_kernel(kernelFIRST,guess_params)        
        second_likelihood=lk.likelihood(second_kernel,x,y,yerr)
            
        for j in range(len(guess_params)):
            prior=np.exp(first_likelihood);
            posterior=np.exp(second_likelihood)
            if prior<1e-300:
                ratio=1
                initial_params[j]=guess_params[j]                
            else:
                ratio = posterior/prior
                if u<np.minimum(1,ratio):
                    initial_params[j]=guess_params[j]
                else:
                    initial_params[j]=initial_params[j]   

            params_list[j].append(initial_params[j])

        first_kernel=new_kernel(kernelFIRST,initial_params)
        first_likelihood=lk.likelihood(first_kernel,x,y,yerr)
        running_logs.append(first_likelihood)
        i+=1
    
    final_kernel=new_kernel(kernelFIRST,initial_params)
    final_likelihood=lk.likelihood(final_kernel,x,y,yerr)
    
    #it returns the final kernel, final likelihood, the evolution of the log
#likelihood and evolution of the parameters of the kernel
    return [final_kernel,final_likelihood,running_logs,params_list]

"""
    new_kernel() updates the parameters of the kernels as the mcmc advances
    
    Parameters
kernelFIRST = original kernel in use
b = new parameters or new hyperparameters if you prefer using that denomination
"""
def new_kernel(kernelFIRST,b): #to update the kernels
    if isinstance(kernelFIRST,kl.ExpSquared):
        return kl.ExpSquared(b[0],b[1])
    elif isinstance(kernelFIRST,kl.ExpSineSquared):
        return kl.ExpSineSquared(b[0],b[1],b[2])
    elif  isinstance(kernelFIRST,kl.RatQuadratic):
        return kl.RatQuadratic(b[0],b[1],b[2])
    elif isinstance(kernelFIRST,kl.Exponential):
        return kl.Exponential(b[0],b[1])
    elif isinstance(kernelFIRST,kl.Matern_32):
        return kl.Matern_32(b[0],b[1])
    elif isinstance(kernelFIRST,kl.Matern_52):
        return kl.Matern_52(b[0],b[1])
    elif isinstance(kernelFIRST,kl.ExpSineGeorge):
        return kl.ExpSineGeorge(b[0],b[1])
    elif isinstance(kernelFIRST,kl.WhiteNoise):
        return kl.WhiteNoise(b[0])
    elif isinstance(kernelFIRST,kl.Sum):
        k1_params=[]
        for i in range(len(kernelFIRST.k1.pars)):
            k1_params.append(b[i])    
        k2_params=[]
        for j in range(len(kernelFIRST.k2.pars)):
            k2_params.append(b[len(kernelFIRST.k1.pars)+j])
        new_k1=new_kernel(kernelFIRST.k1,k1_params)
        new_k2=new_kernel(kernelFIRST.k2,k2_params)
        return new_k1+new_k2
    elif isinstance(kernelFIRST,kl.Product):
        k1_params=[]
        for i in range(len(kernelFIRST.k1.pars)):
            k1_params.append(b[i])    
        k2_params=[]
        for j in range(len(kernelFIRST.k2.pars)):
            k2_params.append(b[len(kernelFIRST.k1.pars)+j])
        new_k1=new_kernel(kernelFIRST.k1,k1_params)
        new_k2=new_kernel(kernelFIRST.k2,k2_params)
        return new_k1*new_k2
    else:
        print 'Something is  missing'