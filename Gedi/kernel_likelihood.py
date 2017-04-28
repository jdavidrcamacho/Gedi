# -*- coding: utf-8 -*-
import kernel as kl
import numpy as np

def build_matrix(kernel, x, y, yerr):
    """
        build_matrix() creates the covariance matrix
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments     
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
    """ 
    from scipy.linalg import cho_factor, cho_solve    
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))    
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)
    n = y.size
    logLike = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike    


def lnlike(K, y):
    """
        lnlike() calculates the marginal log likelihood again?
        Honestly I don't remember why I have it but I keep it here in case its used
    somewhere I'm not remembering. Might get deleted in the future.
    
        Parameters
    K = kernel in use
    y = range of values of te dependent variable (the measurments)    
    """ 
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)
    n = y.size
    logLike = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike


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
    """ 
    from scipy.linalg import cho_factor, cho_solve    
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)

    K_final=K
    
    #Exceptions because of the "white noise diagonal problem"
    if isinstance(kernel, (kl.Sum, kl.Product)):
        if  isinstance(kernel.k1,kl.WhiteNoise):
            oldKernel=kernel
            kernel=kernel.k2
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            K_final=np.vstack([K_final,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = oldKernel(new_r)        
            K_columns = np.vstack([new_lines.T,new_columns])
            K_final=np.hstack([K_final,K_columns])
        
            y_mean=[] #mean = K*.K-1.y  
            for i in range(len(xcalc)):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i in range(len(xcalc)):
                #K**=diag[i]; K*=new_lines[i]  
                a=diag[i]
                newsol = cho_solve(L1, new_lines[i])
                d=np.dot(new_lines[i,:],newsol)
                result=a-d      
                y_var.append(result)  
        if isinstance(kernel.k2,kl.WhiteNoise):
            oldKernel=kernel
            kernel=kernel.k1
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            K_final=np.vstack([K_final,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = oldKernel(new_r)        
            K_columns = np.vstack([new_lines.T,new_columns])
            K_final=np.hstack([K_final,K_columns])
        
            y_mean=[] #mean = K*.K-1.y  
            for i in range(len(xcalc)):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i in range(len(xcalc)):
                #K**=diag[i]; K*=new_lines[i]      
                a=diag[i]
                newsol = cho_solve(L1, new_lines[i])
                d=np.dot(new_lines[i,:],newsol)
                result=a-d      
                y_var.append(result)
        else:
            new_r = xcalc[:, None] - x[None, :]   
            new_lines = kernel(new_r)
            K_final=np.vstack([K_final,new_lines])
            
            new_r = xcalc[:,None] - xcalc[None,:]
            new_columns = kernel(new_r)        
            K_columns = np.vstack([new_lines.T,new_columns])
            K_final=np.hstack([K_final,K_columns])
        
            y_mean=[] #mean = K*.K-1.y  
            for i in range(len(xcalc)):
                y_mean.append(np.dot(new_lines[i,:], sol))
            
            y_var=[] #var=  K** - K*.K-1.K*.T
            diag=np.diagonal(new_columns)
            for i in range(len(xcalc)):
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
        K_final=np.vstack([K_final,new_lines])
        
        new_r = xcalc[:,None] - xcalc[None,:]
        new_columns = kernel(new_r)        
        K_columns = np.vstack([new_lines.T,new_columns])
        K_final=np.hstack([K_final,K_columns])
    
        y_mean=[] #mean = K*.K-1.y  
        for i in range(len(xcalc)):
            y_mean.append(np.dot(new_lines[i,:], sol))
        
        y_var=[] #var=  K** - K*.K-1.K*.T
        diag=np.diagonal(new_columns)
        for i in range(len(xcalc)):
            #K**=diag[i]; K*=new_lines[i]      
            a=diag[i]
            newsol = cho_solve(L1, new_lines[i])
            d=np.dot(new_lines[i,:],newsol)
            result=a-d      
            y_var.append(result)

    y_std = np.sqrt(y_var) #standard deviation
    return [y_mean,y_std]


def likelihood_aux(kernel, x, y, yerr):
    """
        likelihood_aux() makes the covariance matrix calculations necessary
    in the gradient of the log likelihood
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments     
    """
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))
    return K


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
    """ 
    r = x[:, None] - x[None, :]
    K_grad = kernel(r)
    K_inv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(K_inv,y)
    A = np.outer(alpha, alpha) - K_inv
    grad = 0.5 * np.einsum('ij,ij', K_grad, A)
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
    """
    cov_matrix=likelihood_aux(kernel,x,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, y, yerr, cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.dRQ_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dRQ_dalpha,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dRQ_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dExp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dExp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,y,yerr,cov_matrix) 
        return grad1, grad2
    elif isinstance(kernel,kl.Matern32):
        grad1=grad_logp(kernel.dM32_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern52):
        grad1=grad_logp(kernel.dM52_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.WhiteNoise):
        grad1=grad_logp(kernel.dWN_dtheta,x,y,yerr,cov_matrix)
        return grad1
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,y,yerr,cov_matrix) 
        grad_list= [grad1, grad2];      
        return grad_list
    elif isinstance(kernel,kl.Sum):
        grad_list=gradient_sum(kernel,x,y,yerr)
        for i in range(len(grad_list)):
            if isinstance(grad_list[i],float):
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
    """
    kernelOriginal=kernel 
    a=kernel.__dict__
    len_dict=len(kernel.__dict__)
    grad_result=[]    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var] 
        
        if isinstance(k_i,kl.Sum): #to solve the three sums problem
            calc=grad_sum_aux(k_i,x,y,yerr,kernelOriginal)
        else:
            calc=gradient_likelihood_sum(k_i,x,y,yerr,kernelOriginal)
        
        if isinstance(calc, tuple): #to solve the whitenoise problem       
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final =[]
        for j in range(len(grad_result)):         
           grad_final = grad_final + list(grad_result[j])
    return grad_final
    #NoneType -> It might happen if there's no return in gradient_likelihood

           
def gradient_likelihood_sum(kernel,x,y,yerr,kernelOriginal):
    """
        gradient_likelihood_sum() identifies the derivatives to use of a given 
    kernel to calculate the gradient
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    kernelOriginal = original kernel (original sum) being used  
    """ 
    cov_matrix=likelihood_aux(kernelOriginal,x,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, y, yerr, cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.dRQ_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dRQ_dalpha,x,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dRQ_dl,x,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dExp_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dExp_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern32):
        grad1=grad_logp(kernel.dM32_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM32_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern52):
        grad1=grad_logp(kernel.dM52_dtheta,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM52_dl,x,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.WhiteNoise):
        grad1=grad_logp(kernel.dWN_dtheta,x,y,yerr,cov_matrix)       
        return grad1
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,y,yerr,cov_matrix) 
        return grad1, grad2
    elif isinstance(kernel,kl.Product):
        return grad_mul_aux(kernel,x,y,yerr,kernelOriginal)                   
    else:
        print 'gradient -> Something went very wrong!'

 
def grad_sum_aux(kernel,x,y,yerr,kernelOriginal):
    """
        grad_sum_aux() its necesary when we are dealing with multiple sums, i.e. 
    sum of three or more kernels
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    kernelOriginal = original kernel (original sum) being used     
    """ 
    kernelOriginal=kernelOriginal
    a=kernel.__dict__
    len_dict=len(kernel.__dict__)
    grad_result=[]    
    for i in np.arange(1,len_dict+1):
        var = "k{0:d}".format(i)
        k_i = a[var]
        calc=gradient_likelihood_sum(k_i,x,y,yerr,kernelOriginal)
        if isinstance(calc, tuple):        
            grad_result.insert(1,calc)
        else:
            calc=tuple([calc])
            grad_result.insert(1,calc)
        grad_final =[]
        for j in range(len(grad_result)):         
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
    """ 
    kernelOriginal=kernel 
    cov_matrix=likelihood_aux(kernelOriginal,x,y,yerr)

    listofKernels=[kernel.__dict__["k2"]] #to put each kernel separately
    kernel_k1=kernel.__dict__["k1"]

    while len(kernel_k1.__dict__)==2:
        listofKernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listofKernels.insert(0,kernel_k1) #each kernel is now separated
    
    kernelaux1=[];kernelaux2=[]
    for i in range(len(listofKernels)):
        kernelaux1.append(listofKernels[i])
        kernelaux2.append(kernel_deriv(listofKernels[i]))
    
    grad_result=[]
    kernelaux11=kernelaux1;kernelaux22=kernelaux2        
    ii=0    
    while ii<len(listofKernels):    
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels=np.prod(np.array(kernelaux11))
        for ij in range(len(kernelaux22[ii])):
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
    """ 
    if isinstance(kernel,kl.ExpSquared):
        return kernel.dES_dtheta, kernel.dES_dl
    elif isinstance(kernel,kl.ExpSineSquared):
        return kernel.dESS_dtheta, kernel.dESS_dl, kernel.dESS_dP
    elif  isinstance(kernel,kl.RatQuadratic):
        return kernel.dRQ_dtheta, kernel.dRQ_dl, kernel.dRQ_dalpha
    elif isinstance(kernel,kl.Exponential):
        return kernel.dExp_dtheta, kernel.dExp_dl
    elif isinstance(kernel,kl.Matern32):
        return kernel.dM32_dtheta, kernel.dM32_dl
    elif isinstance(kernel,kl.Matern52):
        return kernel.dM52_dtheta, kernel.dM52_dl
    elif isinstance(kernel,kl.ExpSineGeorge):
        return kernel.dE_dGamma, kernel.dE_dP
    elif isinstance(kernel,kl.WhiteNoise):
        return kernel.dWN_dtheta
    else:
        print 'Something went wrong!'
        
              
def grad_mul_aux(kernel,x,y,yerr,kernelOriginal):
    """
        grad_mul_aux() its necesary when we are dealing with multiple terms of 
    sums and multiplications, example: ES*ESS + ES*ESS*WN + RQ*ES*WN and not
    having everything breaking apart
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    kernelOriginal = original kernel (original sum) being used     
    """  
    kernelOriginal=kernelOriginal 
    cov_matrix=likelihood_aux(kernelOriginal,x,y,yerr)

    listofKernels=[kernel.__dict__["k2"]] #to put each kernel separately
    kernel_k1=kernel.__dict__["k1"]

    while len(kernel_k1.__dict__)==2:
        listofKernels.insert(0,kernel_k1.__dict__["k2"])
        kernel_k1=kernel_k1.__dict__["k1"]

    listofKernels.insert(0,kernel_k1) #each kernel is now separated
    
    kernelaux1=[];kernelaux2=[]
    for i in range(len(listofKernels)):
        kernelaux1.append(listofKernels[i])
        kernelaux2.append(kernel_deriv(listofKernels[i]))
    
    grad_result=[]
    kernelaux11=kernelaux1;kernelaux22=kernelaux2        
    ii=0    
    while ii<len(listofKernels):    
        kernelaux11 = kernelaux1[:ii] + kernelaux1[ii+1 :]
        kernels=np.prod(np.array(kernelaux11))
        for ij in range(len(kernelaux22[ii])):
            result=grad_logp(kernelaux2[ii][ij]*kernels,x,y,yerr,cov_matrix)
            grad_result.insert(0,result)
        kernelaux11=kernelaux1;kernelaux22=kernelaux2
        ii=ii+1
        
    grad_result=grad_result[::-1]
    return grad_result   
    
