# -*- coding: utf-8 -*-
import kernel as kl
import kernel_likelihood as lk

import numpy as np
from scipy.linalg import cho_factor, cho_solve 

##### Optimization of the kernels #####
def single_optimization(kernel,x,y,yerr,method='BFGS'):
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
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """
    if method in ["BFGS", "bfgs"]:
        return BFGS(kernel,x,y,yerr)    
    if method in ["SDA", "sda"]:
        return SDA(kernel,x,y,yerr)
    if method in ["RPROP", "rprop"]:
        #this one is questionable
        return RPROP(kernel,x,y,yerr)
    if method in ["altSDA", "altsda"]:
        #I've "invented" this one, I do not guarantee it will work properly
        return altSDA(kernel,x,y,yerr) 



def committed_optimization(kernel,x,y,yerr,max_opt=2,return_method=False):
    """
        commited_optimization() performs the optimization using all algorithms
    and returns the one that gave better results in the end.
        Its slower than the single_optimization() but gives better results. 
    
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    max_opt = optimization runs performed, by default uses 2, recommended upper
            value of 10, more than that it will take a lot of time. 
    return_method = bool to return best optimization method. Default is false.
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """
    i=0
    while i<max_opt:
        log_sda=SDA(kernel,x,y,yerr)
        log_altsda=altSDA(kernel,x,y,yerr)
        log_bfgs=BFGS(kernel,x,y,yerr)        
        logs=[log_sda[0],log_altsda[0],log_bfgs[0]]
        maximum_likelihood=np.max(logs)
        
        if maximum_likelihood==log_sda[0]:
            kernel = log_sda[1]        
        if maximum_likelihood==log_altsda[0]:           
            kernel = log_altsda[1]
        if maximum_likelihood==log_bfgs[0]:           
            kernel = log_bfgs[1]
        i=i+1
    
    logs=[log_sda[0],log_altsda[0],log_bfgs[0]]
    maximum_likelihood=np.max(logs)
    
    if maximum_likelihood==log_sda[0]:
        if return_method:
            return log_sda, "SDA"
        else:
            return log_sda
    if maximum_likelihood==log_altsda[0]:
        if return_method:
            return log_altsda, "altSDA"
        else:
            return log_altsda
    if maximum_likelihood==log_bfgs[0]:
        if return_method:
            return log_bfgs, "BFGS"
        else:    
            return log_bfgs

        
##### Algorithms #####
def BFGS(kernel,x,y,yerr):
    """
        BFGS() is the Broyden Fletcher Goldfarb Shanno Algorithm
        
        Parameters
    kernel = kernel being optimized
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """
    #to not loose the original kernel and data
    original_kernel=kernel;original_x=x
    original_y=y;original_yerr=yerr

    scipystep=1.5e-10 #was 1.4901161193847656e-8 at first 
    step=1e-3 #initial search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0     
    check_it=False
    #we will only start the algorithm when we find the best step to give
    while check_it is False:
        if isinstance(original_kernel,(kl.Sum,kl.Product)):
            hyperparms=[] #initial values of the hyperparam_eters 
            for k, e in enumerate(original_kernel.pars):
                hyperparms.append(original_kernel.pars[k])
            B=np.identity(len(hyperparms)) #Initial matrix   
        else:
            hyperparms=[] #initial values of the hyperparameters 
            for k, e in enumerate(original_kernel.__dict__['pars']):
                hyperparms.append(original_kernel.__dict__['pars'][k])
            B=np.identity(len(hyperparms)) #Initial matrix
            
        #original kernel and gradient    
        first_kernel=new_kernel(original_kernel,hyperparms)
        first_calc= sign_gradlike(first_kernel, original_x,original_y,original_yerr)            
        S1=np.dot(B,first_calc) 
    
        new_hyperparms = [step*n for n in S1] #gives step*S1
        new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                
        #new kernel with hyperparams updated    
        second_kernel=new_kernel(original_kernel,new_hyperparms)
        second_calc=sign_gradlike(second_kernel,original_x,original_y,original_yerr)
    
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i , e in enumerate(second_calc):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)
        if check_it is True: #we are ok to move forward
            step=1.2*step #new bigger step to speedup things
            first_kernel=new_kernel(original_kernel,hyperparms)           
            second_kernel=new_kernel(original_kernel,new_hyperparms) 
        else: #we passed the optimal value and need to go back                    
            step=0.5*step #new smaller step to redo the calculations
            first_kernel=new_kernel(original_kernel,hyperparms) 
        
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
   
    grad_condition=1e-10 #was 1e-3 at first
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        if (it+1)%3!=0:
            check_it=False
            while check_it is False:
                if isinstance(original_kernel,(kl.Sum,kl.Product)):
                    hyperparms=[] #initial values of the hyperparam_eters :
                    for k, e in enumerate(original_kernel.pars):
                        hyperparms.append(original_kernel.pars[k])
                    B=np.identity(len(hyperparms)) #Initial matrix   
                else:
                    hyperparms=[] #initial values of the hyperparameters 
                    for k, e in enumerate(original_kernel.__dict__['pars']):
                        hyperparms.append(original_kernel.__dict__['pars'][k])
                    B=np.identity(len(hyperparms)) #Initial matrix    
                
                #old kernel
                first_kernel=new_kernel(original_kernel,hyperparms)
                first_calc= sign_gradlike(first_kernel,original_x,original_y,original_yerr)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*n for n in S1] #gives step*S1
                new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(original_kernel,new_hyperparms)
                second_calc=sign_gradlike(second_kernel,original_x,original_y,original_yerr)
        
                #lets see if we are going the right direction
                check_sign=[] #to check if we overshot the optimal value           
                for i, e in enumerate(second_calc):
                    check_sign.append(first_calc[i]*second_calc[i])
                check_it=all(check_sign>0 for check_sign in check_sign)
                if check_it is True: #we are ok to move forward
                    step=1.2*step #new bigger step to speedup things
                    first_kernel=new_kernel(original_kernel,hyperparms)           
                    second_kernel=new_kernel(original_kernel,new_hyperparms) 
                else: #we passed the optimal value and need to go back                    
                    step=0.5*step #new smaller step to redo the calculations
                    first_kernel=new_kernel(original_kernel,hyperparms) 
    
            signof_hyperparams=np.min(new_hyperparms)
            if signof_hyperparams<=0:
                second_kernel=first_kernel
                break

            #test of a stoping criteria
            difference=[]
            for i, e in enumerate(first_calc):
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
                if isinstance(original_kernel,(kl.Sum,kl.Product)):
                    hyperparms=[] #initial values of the hyperparam_eters 
                    for k, e in enumerate(original_kernel.pars):
                        hyperparms.append(original_kernel.pars[k])
                    B=np.identity(len(hyperparms)) #Initial matrix   
                else:
                    hyperparms=[] #initial values of the hyperparameters 
                    for k, e in enumerate(original_kernel.__dict__['pars']):
                        hyperparms.append(original_kernel.__dict__['pars'][k])
                    B=np.identity(len(hyperparms)) #Initial matrix
                       
                #old kernel
                first_kernel=new_kernel(original_kernel,hyperparms)
                first_calc= sign_gradlike(first_kernel,original_x,original_y,original_yerr)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*n for n in S1] #gives step*S1
                new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(original_kernel,new_hyperparms)
                second_calc=sign_gradlike(second_kernel,original_x,original_y,original_yerr)
        
                #lets see if we are going the right direction
                check_sign=[] #to check if we overshot the optimal value           
                for i, e in enumerate(second_calc):
                    check_sign.append(first_calc[i]*second_calc[i])
                check_it=all(check_sign>0 for check_sign in check_sign)
                if check_it is True: #we are ok to move forward
                    step=1.2*step #new bigger step to speedup things
                    first_kernel=new_kernel(original_kernel,hyperparms)           
                    second_kernel=new_kernel(original_kernel,new_hyperparms) 
                else: #we passed the optimal value and need to go back                    
                    step=0.5*step #new smaller step to redo the calculations
                    first_kernel=new_kernel(original_kernel,hyperparms) 

            signof_hyperparams=np.min(new_hyperparms)
            if signof_hyperparams<=0:
                second_kernel=first_kernel
                break

            #test of a stoping criteria
            difference=[]
            for i, e in enumerate(first_calc):
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
    final_log= opt_likelihood(second_kernel,original_x,original_y,original_yerr)
    return [final_log,second_kernel]

   
def SDA(kernel,x,y,yerr):
    """
        SDA() is the Steepest descent Algorithm
        
        Parameters
    kernel = kernel being optimized
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """
    original_kernel=kernel;original_x=x
    original_y=y;original_yerr=yerr
    
    scipystep=1.5e-10 #was 1.4901161193847656e-8 at first 
    step=1e-3 #initial search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0
    grad_condition=1e-10 #was 1e-3 at first
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        if isinstance(original_kernel,(kl.Sum,kl.Product)):
            hyperparms=[] #initial values of the hyperparam_eters 
            for k, e in enumerate(original_kernel.pars):
                hyperparms.append(original_kernel.pars[k])
        else:
            hyperparms=[] #initial values of the hyperparameters 
            for k, e in enumerate(original_kernel.__dict__['pars']):
                hyperparms.append(original_kernel.__dict__['pars'][k])        
        
        #to save the 'old' kernel and gradient
        first_kernel=new_kernel(original_kernel,hyperparms)
        first_calc=sign_gradlike(first_kernel, original_x,original_y,original_yerr)

        #update of the hyperparameters
        new_hyperparms = [step*n for n in first_calc]
        new_hyperparms = [n+m for n,m in zip(hyperparms, new_hyperparms)]

        #new kernel with hyperparams updated and gradient
        second_kernel=new_kernel(original_kernel,new_hyperparms) 
        second_calc=sign_gradlike(second_kernel, original_x,original_y,original_yerr)

        signof_hyperparams=np.min(new_hyperparms)
        if signof_hyperparams<=0:
            second_kernel=first_kernel
            break
    
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i ,e in enumerate(second_calc):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)

        if check_it is True: #everything is ok and things can continue                    
            step=1.2*step #new bigger step to speed up the convergence            
            kernel=new_kernel(original_kernel,new_hyperparms) 
        else: #we passed the optimal value and need to go back
            step=0.5*step #new smaller step to redo the calculations
            kernel=new_kernel(original_kernel,hyperparms)        

        #test of a stoping criteria
        difference=[]
        for i ,e in enumerate(first_calc):
            difference.insert(0,abs(second_calc[i]))              
            minimum_difference=np.min(difference)
        minimum_grad=minimum_difference        

        it+=1 #should go back to the start and do the while
        
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,original_x,original_y,original_yerr)
    return [final_log,second_kernel] 

                           
def RPROP(kernel,x,y,yerr):
    """
        RPROP() is the Resilient Propagation Algorithm, 
    I don't trust the results this algorithm gives but still keep it here in 
    the hope of one day make it work
        
        Parameters
    kernel = kernel being optimized
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """ 
    try:
        original_kernel=kernel;original_x=x
        original_y=y;original_yerr=yerr
        
        step=0.005 #initia search step
        dmin=1e-6;dmax=50 #step limits
        minimum_step=1;maximum_step=1 #step difference, 1 to not give error at start
        nplus=1.2;nminus=0.5 #update values of the step
        iterations=200 #maximum number of iterations
        
        
        it=0 #initial iteration
        first_kernel=kernel    
        first_calc=sign_gradlike(kernel, original_x,original_y,original_yerr)
        step_update=[] #steps we will give
        for i ,e in enumerate(first_calc):
            step_update.append(step)
        
        while it<iterations and minimum_step>dmin and maximum_step<dmax:
            hyperparms=[] #initial values of the hyperparameters
            for k, e in enumerate(original_kernel.__dict__['pars']):
                hyperparms.append(first_kernel.__dict__['pars'][k])
                
            new_hyperparms = [sum(n) for n in zip(hyperparms, step_update)]
    
            #new kernel with hyperparams updated
            second_kernel=new_kernel(original_kernel,new_hyperparms)
            second_calc=sign_gradlike(second_kernel, original_x,original_y,original_yerr)
            for j ,e in enumerate(first_calc):
                if first_calc[j]*second_calc[j]>0:
                    step_update[j]=-np.sign(second_calc[i])*step_update[j]*nplus
                    first_kernel=second_kernel        
                    first_calc=second_calc
                    it=it+1
                if first_calc[j]*second_calc[j]<0:
                    step_update[j]=step_update[j]*nminus
                    first_kernel=second_kernel
                    first_calc=()                
                    for i ,e in enumerate(second_calc):
                        first_calc=first_calc+(0,)
                    it=it+1                   
                else:
                    step_update[j]=-np.sign(second_calc[i])*step_update[j]
                    first_kernel=second_kernel        
                    first_calc=second_calc
                    it=it+1
    
            #test of a stoping criteria
            difference=[]
            for i ,e in enumerate(step_update):
                difference.insert(0,abs(step_update[i]))              
                minimum_difference=np.min(difference)
                maximum_difference=np.max(difference)
            minimum_step=minimum_difference    
            maximum_step=maximum_difference
        
        #final likelihood and kernel
        final_log= opt_likelihood(second_kernel,original_x,original_y,original_yerr)        
        return [final_log,second_kernel]        
    except:
        return [-1e10,-1e10]

 
def altSDA(kernel,x,y,yerr):
    """
        altSDA() is the Alternative Steepest descent algorithm I made in my 
    head, combining the properties of the steepest descent algorithm with the 
    rprop algorithm, it work a lot better than what I was expecting.
        
        Parameters
    kernel = kernel being optimized
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments  
    
        Returns
    List with two elements
    List[0] = final log likelihood
    List[1] = final kernel     
    """
    original_kernel=kernel;original_x=x
    original_y=y;original_yerr=yerr
    
    scipystep=1.5e-10 #was 1.4901161193847656e-8 at first 
    step=1e-3 #initia search step
    iterations=1000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start    
    minimum_step=1 #step difference, 1 to not give error at start

    grad_condition=1e-10 #was 1e-3 at first

    it=0
    if isinstance(original_kernel,(kl.Sum,kl.Product)):
        hyperparms=[] #initial values of the hyperparameters 
        for k, e in enumerate(original_kernel.pars):
            hyperparms.append(original_kernel.pars[k]) 
    else:
        hyperparms=[] #initial values of the hyperparameters 
        for k, e in enumerate(original_kernel.__dict__['pars']):
            hyperparms.append(original_kernel.__dict__['pars'][k])

    #initial kernel, gradient, and steps
    first_kernel=new_kernel(original_kernel,hyperparms)
    first_calc=sign_gradlike(first_kernel, original_x,original_y,original_yerr)
    #inital steps we will give for each hyperparameter    
    step_update=list(np.zeros(len(first_calc)))
    for i, e in enumerate(first_calc):
        step_update[i]=step

    while it<iterations and minimum_step>scipystep and minimum_grad>grad_condition:
        #update of the hyperparameters
        new_hyperparms = [n*m for n,m in zip(first_calc,step_update)]
        new_hyperparms = [sum(n) for n in zip(hyperparms, new_hyperparms)]

        #new kernel with hyperparams updated and gradient
        second_kernel=new_kernel(original_kernel,new_hyperparms) 
        second_calc=sign_gradlike(second_kernel, original_x,original_y,original_yerr)

        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value
        final_hyperparameters=[]           
        for i, e in enumerate(second_calc):
            check_sign.append(first_calc[i]*second_calc[i])
            if check_sign[i]>0: #everything is ok and things can continue                    
                step_update[i]=1.2*step_update[i] #new bigger step to speed up the convergence            
                final_hyperparameters.append(new_hyperparms[i])                
            else: #we passed the optimal value and need to go back
                step_update[i]=0.5*step_update[i] #new smaller step to redo the calculations
                final_hyperparameters.append(hyperparms[i])                       

        signof_hyperparams=np.min(new_hyperparms)
        if signof_hyperparams<=0:
            second_kernel=first_kernel
            break
        
        #to update the kernelfor the next iteration       
        hyperparms=final_hyperparameters        
        first_kernel=new_kernel(original_kernel,hyperparms)
        first_calc=sign_gradlike(first_kernel, original_x,original_y,original_yerr)        
            
        #test of a stoping criteria (gradient)
        difference=[]
        for i, e in enumerate(first_calc):
            difference.insert(0,abs(second_calc[i]))              
            minimum_difference=np.min(difference)
        minimum_grad=minimum_difference

        #test of a stoping criteria (step)
        difference=[]
        for i, e in enumerate(step_update):
            difference.insert(0,abs(step_update[i]))              
            minimum_difference=np.min(difference)
        minimum_step=minimum_difference          
        
        it+=1 #should go back to the start and do the while
                
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,original_x,original_y,original_yerr)        
    return [final_log,second_kernel]  

    
##### Auxiliary calculations #####
def opt_likelihood(kernel, x, y, yerr):   
    """
        opt_likelihood() calculates the log likelihood necessary while the
    algorithms make their job, it is not used so it is possible to be removed
    in a future update
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments     
    """ 
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))
    L1 = cho_factor(K)
    sol = cho_solve(L1, y)
    n = y.size
    log_like = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return log_like


def opt_gradlike(kernel, x,y,yerr):
    """
        opt_gradlike() returns the -gradients of the parameters of a kernel
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments     
    """ 
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    grd= [-n for n in grd] #inverts the sign of the gradient
    return grd    


def sign_gradlike(kernel, x,y,yerr):
    """
        sign_gradlike() returns the gradients of the parameters of a kernel
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments     
    """
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    return grd   


def new_kernel(original_kernel,b): #to update the kernels
    """
        new_kernel() updates the parameters of the kernels as the optimizations
    advances
        
        Parameters
    original_kernel = original kernel in use
    b = new parameters or new hyperparameters if you prefer using that denomination
    """
    if isinstance(original_kernel,kl.ExpSquared):
        return kl.ExpSquared(b[0],b[1])
    elif isinstance(original_kernel,kl.ExpSineSquared):
        return kl.ExpSineSquared(b[0],b[1],b[2])
    elif  isinstance(original_kernel,kl.RatQuadratic):
        return kl.RatQuadratic(b[0],b[1],b[2])
    elif isinstance(original_kernel,kl.Exponential):
        return kl.Exponential(b[0],b[1])
    elif isinstance(original_kernel,kl.Matern32):
        return kl.Matern32(b[0],b[1])
    elif isinstance(original_kernel,kl.Matern52):
        return kl.Matern52(b[0],b[1])
    elif isinstance(original_kernel,kl.WhiteNoise):
        return kl.WhiteNoise(b[0])
    elif isinstance(original_kernel,kl.QuasiPeriodic):
        return kl.QuasiPeriodic(b[0],b[1],b[2],b[3])
    elif isinstance(original_kernel,kl.Sum):
        k1_params=[]
        for i, e in enumerate(original_kernel.k1.pars):
            k1_params.append(b[i])    
        k2_params=[]
        for j, e in enumerate(original_kernel.k2.pars):
            k2_params.append(b[len(original_kernel.k1.pars)+j])
        new_k1=new_kernel(original_kernel.k1,k1_params)
        new_k2=new_kernel(original_kernel.k2,k2_params)
        return new_k1+new_k2
    elif isinstance(original_kernel,kl.Product):
        k1_params=[]
        for i, e in enumerate(original_kernel.k1.pars):
            k1_params.append(b[i])    
        k2_params=[]
        for j, e in enumerate(original_kernel.k2.pars):
            k2_params.append(b[len(original_kernel.k1.pars)+j])
        new_k1=new_kernel(original_kernel.k1,k1_params)
        new_k2=new_kernel(original_kernel.k2,k2_params)
        return new_k1*new_k2
    else:
        print 'Something is missing'

##### END