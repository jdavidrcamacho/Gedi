# -*- coding: utf-8 -*-
import Kernel;reload(Kernel);kl = Kernel
import Kernel_likelihood;reload(Kernel_likelihood); lk= Kernel_likelihood

import numpy as np
import inspect
 
##### Optimization of the kernels #####
"""
    single_optimization() allows you to choose what algorithm to use in the
optimization of the kernels

    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
method = algorithm used in the optimization, by default uses BFGS algorithm,
        available algorithms are BFGS, SDA, RPROP and altSDA
""" 
def single_optimization(kernel,x,y,yerr,method='BFGS'):
    if method=='BFGS' or method=='bfgs':
        return BFGS(kernel,x,y,yerr)    
    if method=='SDA' or method=='sda':
        return SDA(kernel,x,y,yerr)
    if method=='RPROP' or method=='rprop':
        #this one is questionable
        return RPROP(kernel,x,y,yerr)
    if method=='altSDA' or method=='altsda':
        #I've "invented" this one, I do not guarantee it will work properly
        return altSDA(kernel,x,y,yerr) 


"""
    commited_optimization() performs the optimization using all algorithms and
returns the one that gave better results in the end.
    Its slower than the single_optimization() but gives better results. 

    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
max_opt = optimization runs performed, by default uses 2, recommended upper
        value of 10, more than that it will take a lot of time. 
""" 
def committed_optimization(kernel,x,y,yerr,max_opt=2):
    i=0
    while i<max_opt:
        log_SDA=SDA(kernel,x,y,yerr)
        log_altSDA=altSDA(kernel,x,y,yerr)
        log_BFGS=BFGS(kernel,x,y,yerr)        
        logs=[log_SDA[0],log_altSDA[0],log_BFGS[0]]
        maximum_likelihood=np.max(logs)
        
        if maximum_likelihood==log_SDA[0]:
            kernel = log_SDA[1]        
        if maximum_likelihood==log_altSDA[0]:           
            kernel = log_altSDA[1]
        if maximum_likelihood==log_BFGS[0]:           
            kernel = log_BFGS[1]
        i=i+1
    
    logs=[log_SDA[0],log_altSDA[0],log_BFGS[0]]
    maximum_likelihood=np.max(logs)
    
    if maximum_likelihood==log_SDA[0]:
        return log_SDA        
    if maximum_likelihood==log_altSDA[0]:
        return log_altSDA
    if maximum_likelihood==log_BFGS[0]:
        return log_BFGS

        
##### Algorithms #####
"""
    BFGS() is the Broyden Fletcher Goldfarb Shanno Algorithm
    
    Parameters
kernel = kernel being optimized
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
"""
def BFGS(kernel,x,y,yerr):
    #to not loose que original kernel and data
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr

    scipystep=1.4901161193847656e-8
    step=1e-3 #initia search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0     
    check_it=False
    #we will only start the algorithm when we find the best step to give
    while check_it is False:
        if isinstance(kernelFIRST,Kernel.Sum) or isinstance(kernelFIRST,Kernel.Product):
            hyperparms=[] #initial values of the hyperparam_eters 
            for k in range(len(kernelFIRST.pars)):
                hyperparms.append(kernelFIRST.pars[k])
            B=np.identity(len(hyperparms)) #Initial matrix   
        else:
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernelFIRST.__dict__['pars'])):
                hyperparms.append(kernelFIRST.__dict__['pars'][k])
            B=np.identity(len(hyperparms)) #Initial matrix
            
        #original kernel and gradient    
        first_kernel=new_kernel(kernelFIRST,hyperparms)
        first_calc= sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)            
        S1=np.dot(B,first_calc) 
    
        new_hyperparms = [step*n for n in S1] #gives step*S1
        new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                
        #new kernel with hyperparams updated    
        second_kernel=new_kernel(kernelFIRST,new_hyperparms)
        second_calc=sign_gradlike(second_kernel,xFIRST,yFIRST,yerrFIRST)
    
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i in range  (len(second_calc)):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)
        if check_it is True: #we are ok to move forward
            step=1.2*step #new bigger step to speedup things
            first_kernel=new_kernel(kernelFIRST,hyperparms)           
            second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
        else: #we passed the optimal value and need to go back                    
            step=0.5*step #new smaller step to redo the calculations
            first_kernel=new_kernel(kernelFIRST,hyperparms) 
        
    #after finding the optimal step we do the calculations of a new matrix B
    d1=np.array([step*n for n in S1]) #gives D1=step*S1
    g1=np.array([n-m for n,m in zip(second_calc,first_calc)])
            
    part1=B #old matrix B
    part2a=np.dot(g1.T,np.dot(B,g1)) #scalar
    part2b=np.dot(d1.T,g1) #this is a scalar
    part2= 1 + part2a/part2b #this is a scalar       
    part3a= np.outer(d1,d1.T) #ths is a matrix
    part3= part3a/part2b #this is a matrix               
    part4a= np.outer(d1,g1.T) #this is a matrix
    part4aa= part4a*part1 #this is a matrix
    part4= part4aa/part2b #this is a matrix  
    part5a= np.outer(g1,d1.T) #this is a matix
    part5b= part1*part5a #this is a matrix
    part5= part5b/part2b #this is a matrix 
    B= part1 + part2*part3 - part4 - part5 #new matrix B
    #To continue we need B, step, and gradient    
   
    grad_condition=1e-3
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        if (it+1)%3!=0:
            check_it=False
            while check_it is False:
                if isinstance(kernelFIRST,Kernel.Sum) or isinstance(kernelFIRST,Kernel.Product):
                    hyperparms=[] #initial values of the hyperparam_eters 
                    for k in range(len(kernelFIRST.pars)):
                        hyperparms.append(kernelFIRST.pars[k])
                    B=np.identity(len(hyperparms)) #Initial matrix   
                else:
                    hyperparms=[] #initial values of the hyperparameters 
                    for k in range(len(kernelFIRST.__dict__['pars'])):
                        hyperparms.append(kernelFIRST.__dict__['pars'][k])
                    B=np.identity(len(hyperparms)) #Initial matrix    
                
                #old kernel
                first_kernel=new_kernel(kernelFIRST,hyperparms)
                first_calc= sign_gradlike(first_kernel,xFIRST,yFIRST,yerrFIRST)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*n for n in S1] #gives step*S1
                new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(kernelFIRST,new_hyperparms)
                second_calc=sign_gradlike(second_kernel,xFIRST,yFIRST,yerrFIRST)
        
                #lets see if we are going the right direction
                check_sign=[] #to check if we overshot the optimal value           
                for i in range  (len(second_calc)):
                    check_sign.append(first_calc[i]*second_calc[i])
                check_it=all(check_sign>0 for check_sign in check_sign)
                if check_it is True: #we are ok to move forward
                    step=1.2*step #new bigger step to speedup things
                    first_kernel=new_kernel(kernelFIRST,hyperparms)           
                    second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
                else: #we passed the optimal value and need to go back                    
                    step=0.5*step #new smaller step to redo the calculations
                    first_kernel=new_kernel(kernelFIRST,hyperparms) 
    
            SignOfHyperparameters=np.min(new_hyperparms)
            if SignOfHyperparameters<=0:
                second_kernel=first_kernel
                break

            #test of a stoping criteria
            difference=[]
            for i  in range(len(first_calc)):
                difference.insert(0,abs(second_calc[i]))              
                minimum_difference=np.min(difference)
            minimum_grad=minimum_difference

            #after finding the optimal step we do the calculations of a new matrix B
            d1=np.array([step*n for n in S1]) #gives D1=step*S1
            g1=np.array([n-m for n,m in zip(second_calc,first_calc)])
                    
            part1=B #old matrix B
            part2a=np.dot(g1.T,np.dot(B,g1)) #scalar
            part2b=np.dot(d1.T,g1) #this is a scalar
            part2= 1 + part2a/part2b #this is a scalar       
            part3a= np.outer(d1,d1.T) #ths is a matrix
            part3= part3a/part2b #this is a matrix               
            part4a= np.outer(d1,g1.T) #this is a matrix
            part4aa= part4a*part1 #this is a matrix
            part4= part4aa/part2b #this is a matrix  
            part5a= np.outer(g1,d1.T) #this is a matix
            part5b= part1*part5a #this is a matrix
            part5= part5b/part2b #this is a matrix 
            B= part1 + part2*part3 - part4 - part5 #new matrix B                
            
        else:
            check_it=False
            while check_it is False:
                if isinstance(kernelFIRST,Kernel.Sum) or isinstance(kernelFIRST,Kernel.Product):
                    hyperparms=[] #initial values of the hyperparam_eters 
                    for k in range(len(kernelFIRST.pars)):
                        hyperparms.append(kernelFIRST.pars[k])
                    B=np.identity(len(hyperparms)) #Initial matrix   
                else:
                    hyperparms=[] #initial values of the hyperparameters 
                    for k in range(len(kernelFIRST.__dict__['pars'])):
                        hyperparms.append(kernelFIRST.__dict__['pars'][k])
                    B=np.identity(len(hyperparms)) #Initial matrix
                       
                #old kernel
                first_kernel=new_kernel(kernelFIRST,hyperparms)
                first_calc= sign_gradlike(first_kernel,xFIRST,yFIRST,yerrFIRST)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*n for n in S1] #gives step*S1
                new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(kernelFIRST,new_hyperparms)
                second_calc=sign_gradlike(second_kernel,xFIRST,yFIRST,yerrFIRST)
        
                #lets see if we are going the right direction
                check_sign=[] #to check if we overshot the optimal value           
                for i in range  (len(second_calc)):
                    check_sign.append(first_calc[i]*second_calc[i])
                check_it=all(check_sign>0 for check_sign in check_sign)
                if check_it is True: #we are ok to move forward
                    step=1.2*step #new bigger step to speedup things
                    first_kernel=new_kernel(kernelFIRST,hyperparms)           
                    second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
                else: #we passed the optimal value and need to go back                    
                    step=0.5*step #new smaller step to redo the calculations
                    first_kernel=new_kernel(kernelFIRST,hyperparms) 

            SignOfHyperparameters=np.min(new_hyperparms)
            if SignOfHyperparameters<=0:
                second_kernel=first_kernel
                break

            #test of a stoping criteria
            difference=[]
            for i  in range(len(first_calc)):
                difference.insert(0,abs(second_calc[i]))              
                minimum_difference=np.min(difference)
            minimum_grad=minimum_difference

            #after finding the optimal step we do the calculations of a new matrix B
            d1=np.array([step*n for n in S1]) #gives D1=step*S1
            g1=np.array([n-m for n,m in zip(second_calc,first_calc)])
                    
            part1=B #old matrix B
            part2a=np.dot(g1.T,np.dot(B,g1)) #scalar
            part2b=np.dot(d1.T,g1) #this is a scalar
            part2= 1 + part2a/part2b #this is a scalar       
            part3a= np.outer(d1,d1.T) #ths is a matrix
            part3= part3a/part2b #this is a matrix               
            part4a= np.outer(d1,g1.T) #this is a matrix
            part4aa= part4a*part1 #this is a matrix
            part4= part4aa/part2b #this is a matrix  
            part5a= np.outer(g1,d1.T) #this is a matix
            part5b= part1*part5a #this is a matrix
            part5= part5b/part2b #this is a matrix 
            B= part1 + part2*part3 - part4 - part5 #new matrix B
            
        it=it+1
    
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,xFIRST,yFIRST,yerrFIRST)
    return [final_log,second_kernel]


"""
    SDA() is the Steepest descent Algorithm
    
    Parameters
kernel = kernel being optimized
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
"""    
def SDA(kernel,x,y,yerr):
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr
    
    scipystep=1.4901161193847656e-8
    step=1e-3 #initia search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0
    grad_condition=1e-3
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        if isinstance(kernelFIRST,Kernel.Sum) or isinstance(kernelFIRST,Kernel.Product):
            hyperparms=[] #initial values of the hyperparam_eters 
            for k in range(len(kernelFIRST.pars)):
                hyperparms.append(kernelFIRST.pars[k])
            B=np.identity(len(hyperparms)) #Initial matrix   
        else:
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernelFIRST.__dict__['pars'])):
                hyperparms.append(kernelFIRST.__dict__['pars'][k])
            B=np.identity(len(hyperparms)) #Initial matrix         
        
        #to save the 'old' kernel and gradient
        first_kernel=new_kernel(kernelFIRST,hyperparms)
        first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)

        #update of the hyperparameters
        new_hyperparms = [step*n for n in first_calc]
        new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]

        #new kernel with hyperparams updated and gradient
        second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
        second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)

        SignOfHyperparameters=np.min(new_hyperparms)
        if SignOfHyperparameters<=0:
            second_kernel=first_kernel
            break
    
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i in range  (len(second_calc)):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)
        #print check_it
        if check_it is True: #everything is ok and things can continue                    
            step=1.2*step #new bigger step to speed up the convergence            
            kernel=new_kernel(kernelFIRST,new_hyperparms) 
        else: #we passed the optimal value and need to go back
            step=0.5*step #new smaller step to redo the calculations
            kernel=new_kernel(kernelFIRST,hyperparms)        

        #test of a stoping criteria
        difference=[]
        for i  in range(len(first_calc)):
            difference.insert(0,abs(second_calc[i]))              
            minimum_difference=np.min(difference)
        minimum_grad=minimum_difference        

        it+=1 #should go back to the start and do the while
        
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,xFIRST,yFIRST,yerrFIRST)
    return [final_log,second_kernel] 


"""
    RPROP() is the Resilient Propagation Algorithm, I don't trust the results
this algorithm gives but still keep it here in the hope of one day make it
work
    
    Parameters
kernel = kernel being optimized
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
"""                            
def RPROP(kernel,x,y,yerr):
    try:
        kernelFIRST=kernel;xFIRST=x
        yFIRST=y;yerrFIRST=yerr
        
        step=0.005 #initia search step
        dmin=1e-6;dmax=50 #step limits
        minimum_step=1;maximum_step=1 #step difference, 1 to not give error at start
        nplus=1.2;nminus=0.5 #update values of the step
        iterations=200 #maximum number of iterations
        
        
        it=0 #initial iteration
        first_kernel=kernel    
        first_calc=sign_gradlike(kernel, xFIRST,yFIRST,yerrFIRST)
        step_update=[] #steps we will give
        for i in range(len(first_calc)):
            step_update.append(step)
        
        while it<iterations and minimum_step>dmin and maximum_step<dmax:
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(first_kernel.__dict__['pars'])):
                hyperparms.append(first_kernel.__dict__['pars'][k])
                
            new_hyperparms = [sum(n) for n in zip(hyperparms, step_update)]
    
            #new kernel with hyperparams updated
            second_kernel=new_kernel(kernelFIRST,new_hyperparms)
            second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)
            for j in range(len(first_calc)):
                if first_calc[j]*second_calc[j]>0:
                    step_update[j]=-np.sign(second_calc[i])*step_update[j]*nplus
                    first_kernel=second_kernel        
                    first_calc=second_calc
                    it=it+1
                if first_calc[j]*second_calc[j]<0:
                    step_update[j]=step_update[j]*nminus
                    first_kernel=second_kernel
                    first_calc=()                
                    for i in range(len(second_calc)):
                        first_calc=first_calc+(0,)
                    it=it+1
                    
                else:
                    step_update[j]=-np.sign(second_calc[i])*step_update[j]
                    first_kernel=second_kernel        
                    first_calc=second_calc
                    it=it+1
    
            #test of a stoping criteria
            difference=[]
            for i  in range(len(step_update)):
                difference.insert(0,abs(step_update[i]))              
                minimum_difference=np.min(difference)
                maximum_difference=np.max(difference)
            minimum_step=minimum_difference    
            maximum_step=maximum_difference
        
        #final likelihood and kernel
        final_log= opt_likelihood(second_kernel,xFIRST,yFIRST,yerrFIRST)        
        return [final_log,second_kernel]        
    except:
        return [-1e10,-1e10]


"""
    altSDA() is the Alternative Steepest descent algorithm I made in my head,
combining the properties of the steepest descent algorithm with the rprop
algorithm, it work a lot better than what I was expecting.
    
    Parameters
kernel = kernel being optimized
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments  
"""  
def altSDA(kernel,x,y,yerr):
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr
    
    scipystep=1.4901161193847656e-8
    step=1e-3 #initia search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start    
    minimum_step=1 #step difference, 1 to not give error at start

    grad_condition=1e-3

    it=0
    if isinstance(kernelFIRST,Kernel.Sum) or isinstance(kernelFIRST,Kernel.Product):
        hyperparms=[] #initial values of the hyperparam_eters 
        for k in range(len(kernelFIRST.pars)):
            hyperparms.append(kernelFIRST.pars[k])
        B=np.identity(len(hyperparms)) #Initial matrix   
    else:
        hyperparms=[] #initial values of the hyperparameters 
        for k in range(len(kernelFIRST.__dict__['pars'])):
            hyperparms.append(kernelFIRST.__dict__['pars'][k])
        B=np.identity(len(hyperparms)) #Initial matrix

    #initial kernel, gradient, and steps
    first_kernel=new_kernel(kernelFIRST,hyperparms)
    first_calc=sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)
    #inital steps we will give for each hyperparameter    
    step_update=list(np.zeros(len(first_calc)))
    for i in range(len(first_calc)):
        step_update[i]=step

    while it<iterations and minimum_step>scipystep and minimum_grad>grad_condition:
        #update of the hyperparameters
        new_hyperparms = [n*m for n,m in zip(first_calc,step_update)]
        new_hyperparms = [sum(n) for n in zip(hyperparms, new_hyperparms)]

        #new kernel with hyperparams updated and gradient
        second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
        second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)

        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value
        final_hyperparameters=[]           
        for i in range  (len(second_calc)):
            check_sign.append(first_calc[i]*second_calc[i])
            if check_sign[i]>0: #everything is ok and things can continue                    
                step_update[i]=1.2*step_update[i] #new bigger step to speed up the convergence            
                final_hyperparameters.append(new_hyperparms[i])                
            else: #we passed the optimal value and need to go back
                step_update[i]=0.5*step_update[i] #new smaller step to redo the calculations
                final_hyperparameters.append(hyperparms[i])                       

        SignOfHyperparameters=np.min(new_hyperparms)
        if SignOfHyperparameters<=0:
            second_kernel=first_kernel
            break
        
        #to update the kernelfor the next iteration       
        hyperparms=final_hyperparameters        
        first_kernel=new_kernel(kernelFIRST,hyperparms)
        first_calc=sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)        
            
        #test of a stoping criteria (gradient)
        difference=[]
        for i  in range(len(first_calc)):
            difference.insert(0,abs(second_calc[i]))              
            minimum_difference=np.min(difference)
        minimum_grad=minimum_difference

        #test of a stoping criteria (step)
        difference=[]
        for i  in range(len(step_update)):
            difference.insert(0,abs(step_update[i]))              
            minimum_difference=np.min(difference)
        minimum_step=minimum_difference          
        
        it+=1 #should go back to the start and do the while
                
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,xFIRST,yFIRST,yerrFIRST)        
    return [final_log,second_kernel]  

    
##### Auxiliary calculations #####
from scipy.linalg import cho_factor, cho_solve
"""
    opt_likelihood() calculates the log likelihood necessary while the
algorithms make their job
    
    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments     
"""      
def opt_likelihood(kernel, x, y, yerr):   
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))       
    log_p_correct = lk.lnlike(K, y)
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)
    n = y.size
    logLike = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike


"""
    opt_gradlike() returns the -gradients of the parameters of a kernel
    
    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments     
""" 
def opt_gradlike(kernel, x,y,yerr):
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    grd= [-grd for grd in grd] #inverts the sign of the gradient
    return grd    


"""
    sign_gradlike() returns the gradients of the parameters of a kernel
    
    Parameters
kernel = kernel in use
x = range of values of the independent variable (usually time)
y = range of values of te dependent variable (the measurments)
yerr = error in the measurments     
"""
def sign_gradlike(kernel, x,y,yerr):
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    return grd   


"""
    new_kernel() updates the parameters of the kernels as the optimizations
advances
    
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
        print 'Something is missing'
