#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve 

from gedi import kernels


def build_matrix(kern, x, yerr):
    """
        build_matrix() creates the covariance matrix

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    K = covariance matrix
    """ 
    r = x[:, None] - x[None, :]
    K = kern(r)
    K = K + yerr**2*np.identity(len(x)) 
    return K


def likelihood(kern, x, y, yerr, kepler = False, kepler_params=[]):
    """
        likelihood() calculates the marginal log likelihood.

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 

    kepler = False if you don't want to use mean function, True otherwise
    kepler_params = [Period, rvAmplitude, ecc, w, t0]
        Returns
    log_like = marginal log likelihood
    """
    if kepler:
        Pk, Krv, e, w, T = kepler_params

        Mean_anom=[2*np.pi*(x1-T)/Pk  for x1 in x] #mean anomaly
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0=[x1 + e*np.sin(x1)  + 0.5*(e**2)*np.sin(2*x1) for x1 in Mean_anom]
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0=[x1 - e*np.sin(x1) for x1 in E0]

        i=0
        while i<100:
            #[x + y for x, y in zip(first, second)]
            calc_aux=[x2-y2 for x2,y2 in zip(Mean_anom,M0)]
            E1=[x3 + y3/(1-e*np.cos(x3)) for x3,y3 in zip(E0,calc_aux)]
            M1=[x4 - e*np.sin(x4) for x4 in E0]
            i+=1
            E0=E1
            M0=M1
        nu=[2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(x5/2)) for x5 in E0]
        RV=[Krv*(e*np.cos(w)+np.cos(w+x6)) for x6 in nu]

        K = build_matrix(kern, x, yerr)
        L1 = cho_factor(K)
        y = np.array(y) - np.array(RV) #to include the keplerian function
        sol = cho_solve(L1, y)
        n = y.size
        log_like = -0.5*np.dot(y, sol) \
                  - np.sum(np.log(np.diag(L1[0]))) \
                  - n*0.5*np.log(2*np.pi)
        return log_like
    else:
        K = build_matrix(kern, x, yerr)
        L1 = cho_factor(K)
        sol = cho_solve(L1, y)
        n = y.size
        log_like = -0.5*np.dot(y, sol) \
                  - np.sum(np.log(np.diag(L1[0]))) \
                  - n*0.5*np.log(2*np.pi)
        return log_like


def minus_likelihood(kernel, t, y, yerr):
    """ Calculates -log_likelihood()
    to be used in scipy.optimize
    """
    return -likelihood(kernel, t, y, yerr)


def new_kernel(original_kernel,b):
    """
        new_kernel() updates the parameters of the kernels as the optimizations
    advances

        Parameters
    original_kernel = original kernel in use
    b = new parameters or new hyperparameters if you prefer using that denomination
    """
    if isinstance(original_kernel,kernels.ExpSquared):
        return kernels.ExpSquared(b[0],b[1])
        
    elif isinstance(original_kernel,kernels.ExpSineSquared):
        return kernels.ExpSineSquared(b[0],b[1],b[2])
        
    elif isinstance(original_kernel,kernels.RatQuadratic):
        return kernels.RatQuadratic(b[0],b[1],b[2])
        
    elif isinstance(original_kernel,kernels.Exponential):
        return kernels.Exponential(b[0],b[1])
        
    elif isinstance(original_kernel,kernels.Matern32):
        return kernels.Matern32(b[0],b[1])
        
    elif isinstance(original_kernel,kernels.Matern52):
        return kernels.Matern52(b[0],b[1])
        
    elif isinstance(original_kernel,kernels.WhiteNoise):
        return kernels.WhiteNoise(b[0])
        
    elif isinstance(original_kernel,kernels.QuasiPeriodic):
        return kernels.QuasiPeriodic(b[0],b[1],b[2],b[3])
        
    elif isinstance(original_kernel,kernels.Sum):
        k1_params = []
        for i, e in enumerate(original_kernel.k1.pars):
            k1_params.append(b[i])    
        k2_params = []
        for j, e in enumerate(original_kernel.k2.pars):
            k2_params.append(b[len(original_kernel.k1.pars)+j])
        new_k1 = new_kernel(original_kernel.k1,k1_params)
        new_k2 = new_kernel(original_kernel.k2,k2_params)
        return new_k1+new_k2
        
    elif isinstance(original_kernel,kernels.Product):
        k1_params = []
        for i, e in enumerate(original_kernel.k1.pars):
            k1_params.append(b[i])    
        k2_params = []
        for j, e in enumerate(original_kernel.k2.pars):
            k2_params.append(b[len(original_kernel.k1.pars)+j])
        new_k1 = new_kernel(original_kernel.k1,k1_params)
        new_k2 = new_kernel(original_kernel.k2,k2_params)
        return new_k1*new_k2
        
    else:
        print('new_kernel: Something is missing')


def gradient_likelihood(kern,x,y,yerr):
    """
        gradient_likelihood() identifies the derivatives to use of a given 
    kernel to calculate the gradient

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments

        Returns
    grad1, grad2, ... = gradients of the kernel derivatives
    """
    cov_matrix = build_matrix(kern,x,yerr)
    if isinstance(kern,kernels.ExpSquared):
        grad1 = grad_lp(kern.des_dtheta, x, y, yerr, cov_matrix)
        grad2 = grad_lp(kern.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2

    elif isinstance(kern,kernels.ExpSineSquared):
        grad1 = grad_lp(kern.dess_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dess_dl,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern,kernels.RatQuadratic):
        grad1 = grad_lp(kern.drq_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.drq_dalpha,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern,kernels.Exponential):
        grad1 = grad_lp(kern.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern,kernels.Matern32):
        grad1 = grad_lp(kern.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern,kernels.Matern52):
        grad1 = grad_lp(kern.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern,kernels.QuasiPeriodic):
        grad1 = grad_lp(kern.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dqp_dl1,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dqp_dl2,x,y,yerr,cov_matrix)
        grad4 = grad_lp(kern.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4

    elif isinstance(kern,kernels.WhiteNoise):
        grad1 = grad_lp(kern.dwn_dtheta,x,y,yerr,cov_matrix)
        return grad1

    elif isinstance(kern,kernels.Sum):
        grad_list = grad_sum(kern,x,y,yerr)
        for i, e in enumerate(grad_list):
            if isinstance(e,float):
                grad_list[i] = [grad_list[i]]
        total = sum(grad_list, [])
        return total

    elif isinstance(kern,kernels.Product):
        return grad_mul(kern,x,y,yerr)

    else:
        print('gradient -> Something went wrong!')


def grad_lp(kern,x,y,yerr,cov_matrix):
    """
        grad_lp() makes the covariance matrix calculations of the kernel
    derivatives and calculates the gradient

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 
    cov_matrix = kernel covariance matrix

        Returns
    See gradient_likelihood(kernel,x,y,yerr) for more info
    """ 
    r = x[:, None] - x[None, :]
    kgrad = kern(r)
    kinv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(kinv,y)
    A = np.outer(alpha, alpha) - kinv
    grad = 0.5 * np.einsum('ij,ij', kgrad, A)
    return grad 


def grad_sum(kern,x,y,yerr):
    """
        grad_sum() makes the gradient calculation for the sums of kernels

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments

        Returns
    See grad_like_sum(kernel,x,y,yerr,original_kernel) for more info
    """
    original_kernel = kern 
    a = kern.__dict__
    len_dict = len(kern.__dict__)
    grad_result = []    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var] 

        if isinstance(k_i,kernels.Sum): #to solve the three sums problem
            calc = grad_sum_aux(k_i,x,y,yerr,original_kernel)
        else:
            calc = grad_like_sum(k_i,x,y,yerr,original_kernel)

        if isinstance(calc, tuple): #to solve the whitenoise problem
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final = []
        for j, e in enumerate(grad_result):

           grad_final = grad_final + list(grad_result[j])
    return grad_final
    #NoneType -> It might happen if there's no return in gradient_likelihood


def grad_like_sum(kern,x,y,yerr,original_kernel):
    """
        grad_like_sum() identifies the derivatives to use of a given 
    kernel to calculate the gradient

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    original_kernel = original kernel (original sum) being used

        Returns
    grad1, grad2, ... = gradients when using a sum operation    
    """ 
    cov_matrix = build_matrix(kern,x,yerr)

    if isinstance(kern, kernels.ExpSquared):
        grad1 = grad_lp(kern.des_dtheta, x, y, yerr, cov_matrix)
        grad2 = grad_lp(kern.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2

    elif isinstance(kern, kernels.ExpSineSquared):
        grad1 = grad_lp(kern.dess_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dess_dl,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern, kernels.RatQuadratic):
        grad1 = grad_lp(kern.drq_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.drq_dalpha,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern, kernels.Exponential):
        grad1 = grad_lp(kern.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, kernels.Matern32):
        grad1 = grad_lp(kern.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, kernels.Matern52):
        grad1 = grad_lp(kern.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, kernels.QuasiPeriodic):
        grad1 = grad_lp(kern.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dqp_dl1,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dqp_dl2,x,y,yerr,cov_matrix)
        grad4 = grad_lp(kern.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4

    elif isinstance(kern, kernels.WhiteNoise):
        grad1 = grad_lp(kern.dwn_dtheta,x,y,yerr,cov_matrix)
        return grad1

    elif isinstance(kern, kernels.Product):
        return grad_mul_aux(kern,x,y,yerr,original_kernel)

    else:
        print('gradient -> Something went very wrong!')


def grad_sum_aux(kern,x,y,yerr,original_kernel):
    """
        grad_sum_aux() its necesary when we are dealing with multiple sums, i.e. 
    sum of three or more kernels

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    original_kernel = original kernel (original sum) being used

        Return
    See grad_like_sum(kernel,x,y,yerr,original_kernel) for more info
    """ 
    original_kernel = original_kernel
    a = kern.__dict__
    len_dict = len(kern.__dict__)
    grad_result = []    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var]
        calc = grad_like_sum(k_i,x,y,yerr,original_kernel)
        if isinstance(calc, tuple):
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final = []
        for j, e in enumerate(grad_result):
           grad_final = grad_final + list(grad_result[j])
    return grad_final


def grad_mul(kern,x,y,yerr):
    """
        grad_mul() makes the gradient calculation of multiplications of 
    kernels 

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments

        Returns
    grad_result = gradients when multiplications are used
    """
    original_kernel = kern 
    cov_matrix = build_matrix(original_kernel,x,yerr)
    listof_kernels = [kern.__dict__["k2"]] #to put each kernel separately
    kernel_k1 = kern.__dict__["k1"]

    while len(kernel_k1.__dict__) == 2:
        listof_kernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listof_kernels.insert(0,kernel_k1) #each kernel is now separated

    kernelaux1=[];kernelaux2 = []
    for i, e in enumerate(listof_kernels):
        kernelaux1.append(listof_kernels[i])
        kernelaux2.append(kernel_deriv(listof_kernels[i]))
    
    grad_result = []
    kernelaux11 = kernelaux1; kernelaux22 = kernelaux2
    ii = 0
    while ii<len(listof_kernels):
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels = np.prod(np.array(kernelaux11))
        for ij, e in enumerate(kernelaux22[ii]):
            result = grad_lp(kernelaux2[ii][ij]*kernels,x,y,yerr,cov_matrix)
            grad_result.insert(0,result)
        kernelaux11 = kernelaux1;kernelaux22=kernelaux2
        ii = ii+1

    grad_result = grad_result[::-1]
    return grad_result 


def kernel_deriv(kern):
    """
        kernel_deriv() identifies the derivatives to use in a given kernel
    
        Parameters
    kern = kernel being use

        Returns
    ... = derivatives of a given kernel
    """ 
    if isinstance(kern, kernels.ExpSquared):
        return kern.des_dtheta, kern.des_dl

    elif isinstance(kern, kernels.ExpSineSquared):
        return kern.dess_dtheta, kern.dess_dl, kern.dess_dp

    elif  isinstance(kern, kernels.RatQuadratic):
        return kern.drq_dtheta, kern.drq_dl, kern.drq_dalpha

    elif isinstance(kern, kernels.Exponential):
        return kern.dexp_dtheta, kern.dexp_dl

    elif isinstance(kern, kernels.Matern32):
        return kern.dm32_dtheta, kern.dm32_dl

    elif isinstance(kern, kernels.Matern52):
        return kern.dm52_dtheta, kern.dm52_dl

    elif isinstance(kern, kernels.WhiteNoise):
        return kern.dwn_dtheta

    elif isinstance(kern, kernels.QuasiPeriodic):
        return kern.dqp_dtheta, kern.dqp_dl1, kern.dqp_dl2, kern.dqp_dp

    else:
        print ('Something went wrong!')


def grad_mul_aux(kern,x,y,yerr,original_kernel):
    """
        grad_mul_aux() its necesary when we are dealing with multiple terms of 
    sums and multiplications, example: ES*ESS + ES*ESS*WN + RQ*ES*WN and not
    having everything breaking apart

        Parameters
    kern = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    original_kernel = original kernel (original sum) being used

        Returns
    See grad_mul(kernel,x,y,yerr) for more info
    """  
    original_kernel = original_kernel 
    cov_matrix = build_matrix(original_kernel,x,yerr)
    listof_kernels = [kern.__dict__["k2"]] #to put each kernel separately
    kernel_k1 = kern.__dict__["k1"]

    while len(kernel_k1.__dict__) == 2:
        listof_kernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listof_kernels.insert(0,kernel_k1) #each kernel is now separated

    kernelaux1 = []; kernelaux2 = []
    for i, e in enumerate(listof_kernels):
        kernelaux1.append(listof_kernels[i])
        kernelaux2.append(kernel_deriv(listof_kernels[i]))

    grad_result = []
    kernelaux11 = kernelaux1; kernelaux22 = kernelaux2
    ii = 0
    while ii<len(listof_kernels):
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels = np.prod(np.array(kernelaux11))
        for ij, e in enumerate(kernelaux22[ii]):
            result = grad_lp(kernelaux2[ii][ij]*kernels,x,y,yerr,cov_matrix)
            grad_result.insert(0,result)
        kernelaux11 = kernelaux1;kernelaux22=kernelaux2
        ii = ii+1

    grad_result = grad_result[::-1]
    return grad_result

##### END
