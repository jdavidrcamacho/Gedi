# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import cho_factor, cho_solve 

from gedi import gpKernel


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

        Returns
    log_like = marginal log likelihood
    """
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
    if isinstance(original_kernel,gpKernel.ExpSquared):
        return gpKernel.ExpSquared(b[0],b[1])
        
    elif isinstance(original_kernel,gpKernel.ExpSineSquared):
        return gpKernel.ExpSineSquared(b[0],b[1],b[2])
        
    elif isinstance(original_kernel,gpKernel.RatQuadratic):
        return gpKernel.RatQuadratic(b[0],b[1],b[2])
        
    elif isinstance(original_kernel,gpKernel.Exponential):
        return gpKernel.Exponential(b[0],b[1])
        
    elif isinstance(original_kernel,gpKernel.Matern32):
        return gpKernel.Matern32(b[0],b[1])
        
    elif isinstance(original_kernel,gpKernel.Matern52):
        return gpKernel.Matern52(b[0],b[1])
        
    elif isinstance(original_kernel,gpKernel.WhiteNoise):
        return gpKernel.WhiteNoise(b[0])
        
    elif isinstance(original_kernel,gpKernel.QuasiPeriodic):
        return gpKernel.QuasiPeriodic(b[0],b[1],b[2],b[3])
        
    elif isinstance(original_kernel,gpKernel.Sum):
        k1_params = []
        for i, e in enumerate(original_kernel.k1.pars):
            k1_params.append(b[i])    
        k2_params = []
        for j, e in enumerate(original_kernel.k2.pars):
            k2_params.append(b[len(original_kernel.k1.pars)+j])
        new_k1 = new_kernel(original_kernel.k1,k1_params)
        new_k2 = new_kernel(original_kernel.k2,k2_params)
        return new_k1+new_k2
        
    elif isinstance(original_kernel,gpKernel.Product):
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
    if isinstance(kern,gpKernel.ExpSquared):
        grad1 = grad_lp(kern.des_dtheta, x, y, yerr, cov_matrix)
        grad2 = grad_lp(kern.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2
        
    elif isinstance(kern,gpKernel.ExpSineSquared):
        grad1 = grad_lp(kern.dess_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dess_dl,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
        
    elif isinstance(kern,gpKernel.RatQuadratic):
        grad1 = grad_lp(kern.drq_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.drq_dalpha,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
        
    elif isinstance(kern,gpKernel.Exponential):
        grad1 = grad_lp(kern.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
        
    elif isinstance(kern,gpKernel.Matern32):
        grad1 = grad_lp(kern.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
        
    elif isinstance(kern,gpKernel.Matern52):
        grad1 = grad_lp(kern.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
        
    elif isinstance(kern,gpKernel.QuasiPeriodic):
        grad1 = grad_lp(kern.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dqp_dl1,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dqp_dl2,x,y,yerr,cov_matrix)
        grad4 = grad_lp(kern.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4
        
    elif isinstance(kern,gpKernel.WhiteNoise):
        grad1 = grad_lp(kern.dwn_dtheta,x,y,yerr,cov_matrix)
        return grad1
        
    elif isinstance(kern,gpKernel.Sum):
        grad_list = grad_sum(kern,x,y,yerr)                
        for i, e in enumerate(grad_list):
            if isinstance(e,float):
                grad_list[i] = [grad_list[i]]
        total = sum(grad_list, [])
        return total
        
    elif isinstance(kern,gpKernel.Product):
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
        
        if isinstance(k_i,gpKernel.Sum): #to solve the three sums problem
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
    
    if isinstance(kern, gpKernel.ExpSquared):
        grad1 = grad_lp(kern.des_dtheta, x, y, yerr, cov_matrix)
        grad2 = grad_lp(kern.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2

    elif isinstance(kern, gpKernel.ExpSineSquared):
        grad1 = grad_lp(kern.dess_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dess_dl,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern, gpKernel.RatQuadratic):
        grad1 = grad_lp(kern.drq_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.drq_dalpha,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 

    elif isinstance(kern, gpKernel.Exponential):
        grad1 = grad_lp(kern.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, gpKernel.Matern32):
        grad1 = grad_lp(kern.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, gpKernel.Matern52):
        grad1 = grad_lp(kern.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2

    elif isinstance(kern, gpKernel.QuasiPeriodic):
        grad1 = grad_lp(kern.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2 = grad_lp(kern.dqp_dl1,x,y,yerr,cov_matrix)
        grad3 = grad_lp(kern.dqp_dl2,x,y,yerr,cov_matrix)
        grad4 = grad_lp(kern.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4

    elif isinstance(kern, gpKernel.WhiteNoise):
        grad1 = grad_lp(kern.dwn_dtheta,x,y,yerr,cov_matrix)       
        return grad1

    elif isinstance(kern, gpKernel.Product):
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
    if isinstance(kern, gpKernel.ExpSquared):
        return kern.des_dtheta, kern.des_dl
        
    elif isinstance(kern, gpKernel.ExpSineSquared):
        return kern.dess_dtheta, kern.dess_dl, kern.dess_dp

    elif  isinstance(kern, gpKernel.RatQuadratic):
        return kern.drq_dtheta, kern.drq_dl, kern.drq_dalpha

    elif isinstance(kern, gpKernel.Exponential):
        return kern.dexp_dtheta, kern.dexp_dl

    elif isinstance(kern, gpKernel.Matern32):
        return kern.dm32_dtheta, kern.dm32_dl

    elif isinstance(kern, gpKernel.Matern52):
        return kern.dm52_dtheta, kern.dm52_dl

    elif isinstance(kern, gpKernel.WhiteNoise):
        return kern.dwn_dtheta

    elif isinstance(kern, gpKernel.QuasiPeriodic):
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
