# -*- coding: utf-8 -*-
import kernel as kl
import numpy as np
from scipy.linalg import cho_factor, cho_solve 

def build_matrix(kernel, x, y, yerr):
    """
        build_matrix() creates the covariance matrix
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    K = covariance matrix
    """ 
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x)) 
    return K


def likelihood(kernel, x, y, yerr):    
    """
        likelihood() calculates the marginal log likelihood.
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 

        Returns
    log_like = marginal log likelihood
    """
    K=build_matrix(kernel, x, y, yerr)    
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)
    n = y.size
    log_like = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return log_like    


def compute_kernel(kernel, x, xcalc, y, yerr):    
    """
        compute_kenrel() makes the necessary calculations to allow the user to 
    create pretty graphics in the end, the ones that includes the mean and 
    standard deviation. 
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    xcalc = new range of values to calculate the means and standard deviation, in
            other words, to predict value of the kernel between measurments, as
            such we should have xcalc >> x
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    [y_mean,y_std] = [mean, standard deviation]
    """
    K=build_matrix(kernel, x, y, yerr)    
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)

    kfinal=K

    #Exceptions because of the "white noise diagonal problem"
    if isinstance(kernel, (kl.Sum, kl.Product)):
        if  isinstance(kernel.k1,kl.WhiteNoise):
            old_kernel=kernel
            kernel=kernel.k2
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            kfinal=np.vstack([kfinal,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = old_kernel(new_r)        
            kcolumns = np.vstack([new_lines.T,new_columns])
            kfinal=np.hstack([kfinal,kcolumns])
        
            y_mean=[] #mean = K*.K-1.y  :
            for i, e in enumerate(xcalc):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i, e in enumerate(xcalc):
                #K**=diag[i]; K*=new_lines[i]  
                a=diag[i]
                newsol = cho_solve(L1, new_lines[i])
                d=np.dot(new_lines[i,:],newsol)
                result=a-d      
                y_var.append(result)  
        if isinstance(kernel.k2,kl.WhiteNoise):
            old_kernel=kernel
            kernel=kernel.k1
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            kfinal=np.vstack([kfinal,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = old_kernel(new_r)        
            kcolumns = np.vstack([new_lines.T,new_columns])
            kfinal=np.hstack([kfinal,kcolumns])
        
            y_mean=[] #mean = K*.K-1.y  
            for i, e in enumerate(xcalc):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i, e in enumerate(xcalc):
                #K**=diag[i]; K*=new_lines[i]      
                a=diag[i]
                newsol = cho_solve(L1, new_lines[i])
                d=np.dot(new_lines[i,:],newsol)
                result=a-d      
                y_var.append(result)
        else:
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            kfinal=np.vstack([kfinal,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = kernel(new_r)        
            kcolumns = np.vstack([new_lines.T,new_columns])
            kfinal=np.hstack([kfinal,kcolumns])
        
            y_mean=[] #mean = K*.K-1.y  
            for i, e in enumerate(xcalc):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i, e in enumerate(xcalc):
                #K**=diag[i]; K*=new_lines[i]      
                a=diag[i]
                newsol = cho_solve(L1, new_lines[i])
                d=np.dot(new_lines[i,:],newsol)
                result=a-d      
                y_var.append(result)
    
    #If we are not using a white noise kernel things are ok to continue
    else:    
        new_r = xcalc[:, None] - x[None, :]   
        new_lines = kernel(new_r)
        kfinal=np.vstack([kfinal,new_lines])
        
        new_r = xcalc[:,None] - xcalc[None,:]
        new_columns = kernel(new_r)        
        kcolumns = np.vstack([new_lines.T,new_columns])
        kfinal=np.hstack([kfinal,kcolumns])
    
        y_mean=[] #mean = K*.K-1.y  
        for i, e in enumerate(xcalc):
            y_mean.append(np.dot(new_lines[i,:], sol))
        
        y_var=[] #var=  K** - K*.K-1.K*.T
        diag=np.diagonal(new_columns)
        for i, e in enumerate(xcalc):
            #K**=diag[i]; K*=new_lines[i]      
            a=diag[i]
            newsol = cho_solve(L1, new_lines[i])
            d=np.dot(new_lines[i,:],newsol)
            result=a-d      
            y_var.append(result)

    y_std = np.sqrt(y_var) #standard deviation
    return [y_mean,y_std]


def grad_logp(kernel,x,y,yerr,cov_matrix):
    """
        grad_logp() makes the covariance matrix calculations of the kernel
    derivatives and calculates the gradient
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 
    cov_matrix = kernel covariance matrix
    
        Returns
    See gradient_likelihood(kernel,x,y,yerr) for more info
    """ 
    r = x[:, None] - x[None, :]
    kgrad = kernel(r)
    kinv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(kinv,y)
    A = np.outer(alpha, alpha) - kinv
    grad = 0.5 * np.einsum('ij,ij', kgrad, A)
    return grad 


def gradient_likelihood(kernel,x,y,yerr):
    """
        gradient_likelihood() identifies the derivatives to use of a given 
    kernel to calculate the gradient
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    grad1, grad2, ... = gradients of the kernel derivatives
    """
    cov_matrix=build_matrix(kernel,x,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.des_dtheta, x, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dess_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dess_dl,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.drq_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.drq_dalpha,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern32):
        grad1=grad_logp(kernel.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern52):
        grad1=grad_logp(kernel.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.QuasiPeriodic):
        grad1=grad_logp(kernel.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dqp_dl1,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dqp_dl2,x,y,yerr,cov_matrix)
        grad4=grad_logp(kernel.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4
    elif isinstance(kernel,kl.WhiteNoise):
        grad1=grad_logp(kernel.dwn_dtheta,x,y,yerr,cov_matrix)
        return grad1
    elif isinstance(kernel,kl.Sum):
        grad_list=gradient_sum(kernel,x,y,yerr)                
        for i, e in enumerate(grad_list):
            if isinstance(e,float):
                grad_list[i]=[grad_list[i]]
        total=sum(grad_list, [])
        return total        
    elif isinstance(kernel,kl.Product):
        return gradient_mul(kernel,x,y,yerr)                
    else:
        print 'gradient -> Something went wrong!'

    
def gradient_sum(kernel,x,y,yerr):
    """
        gradient_sum() makes the gradient calculation for the sums of kernels
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    See gradient_likelihood_sum(kernel,x,y,yerr,original_kernel) for more info
    """
    original_kernel=kernel 
    a=kernel.__dict__
    len_dict=len(kernel.__dict__)
    grad_result=[]    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var] 
        
        if isinstance(k_i,kl.Sum): #to solve the three sums problem
            calc=grad_sum_aux(k_i,x,y,yerr,original_kernel)
        else:
            calc=gradient_likelihood_sum(k_i,x,y,yerr,original_kernel)
        
        if isinstance(calc, tuple): #to solve the whitenoise problem       
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final =[]
        for j, e in enumerate(grad_result):            
            
           grad_final = grad_final + list(grad_result[j])
    return grad_final
    #NoneType -> It might happen if there's no return in gradient_likelihood

           
def gradient_likelihood_sum(kernel,x,y,yerr,original_kernel):
    """
        gradient_likelihood_sum() identifies the derivatives to use of a given 
    kernel to calculate the gradient
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    original_kernel = original kernel (original sum) being used
    
        Returns
    grad1, grad2, ... = gradients when using a sum operation    
    """ 
#    cov_matrix=build_matrix(original_kernel,x,y,yerr)
    cov_matrix=build_matrix(kernel,x,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.des_dtheta, x, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.des_dl, x, y, yerr, cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dess_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dess_dl,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dess_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.drq_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.drq_dalpha,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.drq_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dexp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dexp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern32):
        grad1=grad_logp(kernel.dm32_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dm32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern52):
        grad1=grad_logp(kernel.dm52_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dm52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.QuasiPeriodic):
        grad1=grad_logp(kernel.dqp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dqp_dl1,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dqp_dl2,x,y,yerr,cov_matrix)
        grad4=grad_logp(kernel.dqp_dp,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3, grad4
    elif isinstance(kernel,kl.WhiteNoise):
        grad1=grad_logp(kernel.dwn_dtheta,x,y,yerr,cov_matrix)       
        return grad1
    elif isinstance(kernel,kl.Product):
        return grad_mul_aux(kernel,x,y,yerr,original_kernel)                   
    else:
        print 'gradient -> Something went very wrong!'

 
def grad_sum_aux(kernel,x,y,yerr,original_kernel):
    """
        grad_sum_aux() its necesary when we are dealing with multiple sums, i.e. 
    sum of three or more kernels
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    original_kernel = original kernel (original sum) being used

        Return
    See gradient_likelihood_sum(kernel,x,y,yerr,original_kernel) for more info
    """ 
    original_kernel=original_kernel
    a=kernel.__dict__
    len_dict=len(kernel.__dict__)
    grad_result=[]    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var]
        calc=gradient_likelihood_sum(k_i,x,y,yerr,original_kernel)
        if isinstance(calc, tuple):        
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final =[]
        for j, e in enumerate(grad_result):
           grad_final = grad_final + list(grad_result[j])     
    return grad_final
    
         
def gradient_mul(kernel,x,y,yerr):
    """
        gradient_mul() makes the gradient calculation of multiplications of 
    kernels 
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    grad_result = gradients when multiplications are used
    """ 
    original_kernel=kernel 
    cov_matrix=build_matrix(original_kernel,x,y,yerr)
    listof_kernels=[kernel.__dict__["k2"]] #to put each kernel separately
    kernel_k1=kernel.__dict__["k1"]

    while len(kernel_k1.__dict__)==2:
        listof_kernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listof_kernels.insert(0,kernel_k1) #each kernel is now separated
    
    kernelaux1=[];kernelaux2=[]
    for i, e in enumerate(listof_kernels):
        kernelaux1.append(listof_kernels[i])
        kernelaux2.append(kernel_deriv(listof_kernels[i]))
    
    grad_result=[]
    kernelaux11=kernelaux1;kernelaux22=kernelaux2        
    ii=0    
    while ii<len(listof_kernels):    
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels=np.prod(np.array(kernelaux11))
        for ij, e in enumerate(kernelaux22[ii]):
            result=grad_logp(kernelaux2[ii][ij]*kernels,x,y,yerr,cov_matrix)
            grad_result.insert(0,result)
        kernelaux11=kernelaux1;kernelaux22=kernelaux2
        ii=ii+1
        
    grad_result=grad_result[::-1]
    return grad_result 

           
def kernel_deriv(kernel):
    """
        kernel_deriv() identifies the derivatives to use in a given kernel
    
        Parameters
    kernel = kernel being use
    
        Returns
    ... = derivatives of a given kernel
    """ 
    if isinstance(kernel,kl.ExpSquared):
        return kernel.des_dtheta, kernel.des_dl
    elif isinstance(kernel,kl.ExpSineSquared):
        return kernel.dess_dtheta, kernel.dess_dl, kernel.dess_dp
    elif  isinstance(kernel,kl.RatQuadratic):
        return kernel.drq_dtheta, kernel.drq_dl, kernel.drq_dalpha
    elif isinstance(kernel,kl.Exponential):
        return kernel.dexp_dtheta, kernel.dexp_dl
    elif isinstance(kernel,kl.Matern32):
        return kernel.dm32_dtheta, kernel.dm32_dl
    elif isinstance(kernel,kl.Matern52):
        return kernel.dm52_dtheta, kernel.dm52_dl
    elif isinstance(kernel,kl.WhiteNoise):
        return kernel.dwn_dtheta
    elif isinstance(kernel,kl.QuasiPeriodic):
        return kernel.dqp_dtheta, kernel.dqp_dl1, kernel.dqp_dl2, kernel.dqp_dp
    else:
        print 'Something went wrong!'
         
              
def grad_mul_aux(kernel,x,y,yerr,original_kernel):
    """
        grad_mul_aux() its necesary when we are dealing with multiple terms of 
    sums and multiplications, example: ES*ESS + ES*ESS*WN + RQ*ES*WN and not
    having everything breaking apart
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    original_kernel = original kernel (original sum) being used
    
        Returns
    See gradient_mul(kernel,x,y,yerr) for more info
    """  
    original_kernel=original_kernel 
    cov_matrix= build_matrix(original_kernel,x,y,yerr)
    listof_kernels=[kernel.__dict__["k2"]] #to put each kernel separately
    kernel_k1=kernel.__dict__["k1"]

    while len(kernel_k1.__dict__)==2:
        listof_kernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listof_kernels.insert(0,kernel_k1) #each kernel is now separated
    
    kernelaux1=[];kernelaux2=[]
    for i, e in enumerate(listof_kernels):       
        kernelaux1.append(listof_kernels[i])
        kernelaux2.append(kernel_deriv(listof_kernels[i]))
    
    grad_result=[]
    kernelaux11=kernelaux1;kernelaux22=kernelaux2        
    ii=0    
    while ii<len(listof_kernels):    
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels=np.prod(np.array(kernelaux11))
        for ij, e in enumerate(kernelaux22[ii]):
            result=grad_logp(kernelaux2[ii][ij]*kernels,x,y,yerr,cov_matrix)
            grad_result.insert(0,result)
        kernelaux11=kernelaux1;kernelaux22=kernelaux2
        ii=ii+1
        
    grad_result=grad_result[::-1]
    return grad_result   
    
##### END