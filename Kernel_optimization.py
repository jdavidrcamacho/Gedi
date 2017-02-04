# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:40:03 2017

@author: camacho
"""

import Kernel;reload(Kernel);kl = Kernel
import Kernel_likelihood;reload(Kernel_likelihood); lk= Kernel_likelihood

import numpy as np
import inspect
 
##### OPTIMIZATION ############################################################

##### Start optimization
def optimization(kernel,x,y,yerr,method='BFGS'):
    if method=='BFGS' or method=='bfgs':
        BFGS(kernel,x,y,yerr)    
    if method=='SDA' or method=='sda':
        SDA(kernel,x,y,yerr)
    if method=='RPROP' or method=='rprop':
        RPROP(kernel,x,y,yerr) #this one is questionable

    
##### algorithms
scipystep=1.4901161193847656e-08 #taken from scipy.optimize

### BFGS - Broyden Fletcher Goldfarb Shanno Algorithm
def BFGS(kernel,x,y,yerr):
    #to not loose que original kernel and data
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr
    
    step=0.005 #initia search step
    iterations=2000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0     
    check_it=False
    #we will only start the algorithm when we find the best step to give
    while check_it is False: 
        hyperparms=[] #initial values of the hyperparameters 
        for k in range(len(kernelFIRST.__dict__['pars'])):
            hyperparms.append(kernelFIRST.__dict__['pars'][k])
        B=np.identity(len(hyperparms)) #Initial matrix
    
        #original kernel and gradient    
        first_kernel=new_kernel(kernelFIRST,hyperparms)
        first_calc= sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)       
        S1=np.dot(B,first_calc) 
    
        new_hyperparms = [step*x for x in S1] #gives step*S1
        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
                
        #new kernel with hyperparams updated    
        second_kernel=new_kernel(kernelFIRST,new_hyperparms)
        second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)
  
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i in range  (len(second_calc)):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)
        if check_it is True: #we are ok to move forward
            first_kernel=new_kernel(kernelFIRST,hyperparms)           
            second_kernel=new_kernel(kernelFIRST,new_hyperparms) 
        else: #we passed the optimal value and need to go back                    
            step=0.5*step #new smaller step to redo the calculations
            first_kernel=new_kernel(kernelFIRST,hyperparms) 
    
    #after finding the optimal step we do the calculations of a new matrix B
    d1=np.array([step*x for x in S1]) #gives D1=step*S1
    g1=np.array([a-b for a,b in zip(second_calc,first_calc)])
            
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
    
    grad_condition=0.01
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        if (it+1)%3!=0:
            check_it=False
            while check_it is False:
                hyperparms=[] #initial values of the hyperparameters 
                for k in range(len(second_kernel.__dict__['pars'])):
                    hyperparms.append(second_kernel.__dict__['pars'][k])        
                
                #old kernel
                first_kernel=new_kernel(kernelFIRST,hyperparms)
                first_calc= sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*x for x in S1] #gives step*S1
                new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(kernelFIRST,new_hyperparms)
                second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)
        
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

            #test of a stoping criteria
            difference=[]
            for i  in range(len(first_calc)):
                difference.insert(0,abs(second_calc[i]))              
                minimum_difference=np.min(difference)
            minimum_grad=minimum_difference

            #after finding the optimal step we do the calculations of a new matrix B
            d1=np.array([step*x for x in S1]) #gives D1=step*S1
            g1=np.array([a-b for a,b in zip(second_calc,first_calc)])
                    
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
            
            SignOfHyperparameters=np.min(new_hyperparms)
            if SignOfHyperparameters<=0:
                print 'ERROR! Check the initial hyperparameters'
                print 'Bad initial hyperparameters influences the result'
                break
        else:
            check_it=False
            while check_it is False:
                hyperparms=[] #initial values of the hyperparameters 
                for k in range(len(second_kernel.__dict__['pars'])):
                    hyperparms.append(second_kernel.__dict__['pars'][k])        
                B=np.identity(len(hyperparms)) #reset of matrix B 
               
                #old kernel
                first_kernel=new_kernel(kernelFIRST,hyperparms)
                first_calc= sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)              
                S1=np.dot(B,first_calc) #New S1 
                new_hyperparms = [step*x for x in S1] #gives step*S1
                new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
                #new kernel with hyperparams updated    
                second_kernel=new_kernel(kernelFIRST,new_hyperparms)
                second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)
        
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

            #test of a stoping criteria
            difference=[]
            for i  in range(len(first_calc)):
                difference.insert(0,abs(second_calc[i]))              
                minimum_difference=np.min(difference)
            minimum_grad=minimum_difference

            #after finding the optimal step we do the calculations of a new matrix B
            d1=np.array([step*x for x in S1]) #gives D1=step*S1
            g1=np.array([a-b for a,b in zip(second_calc,first_calc)])
                    
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

            SignOfHyperparameters=np.min(new_hyperparms)
            if SignOfHyperparameters<=0:
                print 'ERROR! Check the initial hyperparameters'
                print 'Bad initial hyperparameters influences the result'
                break
            
        it=it+1
    
    #final likelihood and kernel
    final_log= opt_likelihood(second_kernel,xFIRST,yFIRST,yerrFIRST)    
    print 'total iterations ->', it
    print 'final log likelihood ->', final_log
    print 'final kernel ->', second_kernel  

###  SDA - Steepest descent Algorithm
def SDA(kernel,x,y,yerr):
    #to not loose que original kernel and data
    kernelFIRST=kernel;xFIRST=x
    yFIRST=y;yerrFIRST=yerr
    
    step=0.005 #initia search step
    iterations=2000 #maximum number of iterations
    minimum_grad=1 #gradient difference, 1 to not give error at start
    
    it=0
    grad_condition=0.001
    while it<iterations and step>scipystep and minimum_grad>grad_condition:
        hyperparms=[] #initial values of the hyperparameters 
        for k in range(len(kernel.__dict__['pars'])):
            hyperparms.append(kernel.__dict__['pars'][k])            
        
        #to save the 'old' kernel and gradient
        first_kernel=new_kernel(kernelFIRST,hyperparms)
        first_calc=sign_gradlike(first_kernel, xFIRST,yFIRST,yerrFIRST)

        #update of the hyperparameters
        new_hyperparms = [step*x for x in first_calc]
        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
        kernel.__dict__['pars'][:]=new_hyperparms 
        a = kernel.__dict__['pars']
        b=[]    
        for ij in range(len(a)):
            b.append(a[ij])         
        
        #new kernel with hyperparams updated and gradient
        second_kernel=new_kernel(kernelFIRST,b) 
        second_calc=sign_gradlike(second_kernel, xFIRST,yFIRST,yerrFIRST)
        
        #lets see if we are going the right direction
        check_sign=[] #to check if we overshot the optimal value           
        for i in range  (len(second_calc)):
            check_sign.append(first_calc[i]*second_calc[i])
        check_it=all(check_sign>0 for check_sign in check_sign)
        #print check_it
        if check_it is True: #everything is ok and things can continue                    
            step=1.2*step #new bigger step to speed up the convergence            
            kernel=new_kernel(kernelFIRST,b) 
        else: #we passed the optimal value and need to go back
            step=0.5*step #new smaller step to redo the calculations
            kernel=new_kernel(kernelFIRST,hyperparms)        
        it+=1 #should go back to the start and do the while
        
        #test of a stoping criteria
        difference=[]
        for i  in range(len(first_calc)):
            difference.insert(0,abs(second_calc[i]))              
            minimum_difference=np.min(difference)
        minimum_grad=minimum_difference        
                
    #final likelihood and kernel
    final_log= opt_likelihood(kernel,xFIRST,yFIRST,yerrFIRST)        
    print 'total iterations ->', it
    print 'final log likelihood ->', final_log
    print 'final kernel ->', second_kernel  
                                
### RPROP - Resilient Propagation Algorithm
#I don't trust this algorithm but still keep it here
def RPROP(kernel,x,y,yerr):
    #to not loose que original kernel and data
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
        #print 'steps',step_update
        hyperparms=[] #initial values of the hyperparameters 
        for k in range(len(first_kernel.__dict__['pars'])):
            hyperparms.append(first_kernel.__dict__['pars'][k])
            
        new_hyperparms = [sum(x) for x in zip(hyperparms, step_update)]

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
    print 'total iterations ->', it
    print 'final log likelihood ->',  final_log
    print 'final kernel ->', second_kernel

    
##### Auxiliary calculations ##################################################       
def opt_likelihood(kernel, x, y, yerr): #covariance matrix calculations   
    r = x[:, None] - x[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x))       
    log_p_correct = lk.lnlike(K, y)
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, y) # this is K^-1*(r)
    n = y.size
    logLike = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike

def opt_gradlike(kernel, x,y,yerr):
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    grd= [-grd for grd in grd] #inverts the sign of the gradient
    return grd    

def sign_gradlike(kernel, x,y,yerr):
    grd= lk.gradient_likelihood(kernel, x,y,yerr) #gradient likelihood
    return grd   

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
    else:
        print 'Falta algo'


##### Old/Experimental Algorithms #############################################        
#### altSDA - Alternative Steepest descent algorithm I made in my head
#def altSDA(kernel,x,xcalc,y,yerr):
#    #to not loose que original kernel and data
#    kernelFIRST=kernel
#    xFIRST=x;xcalcFIRST=xcalc
#    yFIRST=y;yerrFIRST=yerr
#    
#    step=0.005 #initia search step
#    iterations=2000 #maximum number of iterations
#    minimum_grad=1 #gradient difference, 1 to not give error at start
#    
#    it=0
#    grad_condition=0.001
#
#    #initial kernel, gradient, and steps
#    first_kernel=new_kernel(kernelFIRST,hyperparms)
#    first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#    #inital steps we will give for each hyperparameter    
#    step_update=[]
#    for i in range(len(first_calc)):
#        step_update.append(step)
#    
#    while it<iterations and step>scipystep and minimum_grad>grad_condition:
#        hyperparms=[] #initial values of the hyperparameters 
#        for k in range(len(kernel.__dict__['pars'])):
#            hyperparms.append(kernel.__dict__['pars'][k])            
#        
#        #to save the 'old' kernel and gradient
#        first_kernel=new_kernel(kernelFIRST,hyperparms)
#        first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#        
#        for i in range(len(first_calc)):
#            
#
#        #update of the hyperparameters
#        new_hyperparms = [step*x for x in first_calc]
#        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
#        kernel.__dict__['pars'][:]=new_hyperparms 
#        a = kernel.__dict__['pars']
#        b=[]    
#        for ij in range(len(a)):
#            b.append(a[ij])         
#        
#        #new kernel with hyperparams updated and gradient
#        second_kernel=new_kernel(kernelFIRST,b) 
#        second_calc=sign_gradlike(second_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#        
#        #lets see if we are going the right direction
#        check_sign=[] #to check if we overshot the optimal value           
#        for i in range  (len(second_calc)):
#            check_sign.append(first_calc[i]*second_calc[i])
#        check_it=all(check_sign>0 for check_sign in check_sign)
#        #print check_it
#        if check_it is True: #everything is ok and things can continue                    
#            step=1.2*step #new bigger step to speed up the convergence            
#            kernel=new_kernel(kernelFIRST,b) 
#        else: #we passed the optimal value and need to go back
#            step=0.5*step #new smaller step to redo the calculations
#            kernel=new_kernel(kernelFIRST,hyperparms)        
#        it+=1 #should go back to the start and do the while
#        
#        #test of a stoping criteria
#        difference=[]
#        for i  in range(len(first_calc)):
#            difference.insert(0,abs(second_calc[i]))              
#            minimum_difference=np.min(difference)
#        minimum_grad=minimum_difference        
#                
#    #final likelihood and kernel
#    final_log= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST)        
#    print 'total iterations ->', it
#    print 'final log likelihood ->', final_log
#    print 'final kernel ->', second_kernel
        
#### CGA - Conjugate gradient (Fletcher-Reeves) Algorithm
#def CGA(kernel,x,xcalc,y,yerr):
#    #not to loose que original kernel and data
#    kernelFIRST=kernel
#    xFIRST=x;xcalcFIRST=xcalc
#    yFIRST=y;yerrFIRST=yerr
#
#    step=0.005 #initia search step
#    iterations=2000 #maximum number of iterations
#    minimum_grad=1 #gradient difference, 1 to not give error at start
#    
#    it=0
#
#    #we will only start the algorithm when we find the best step to give
#    check_it=False
#    while check_it is False: 
#        hyperparms=[] #initial values of the hyperparameters 
#        for k in range(len(kernelFIRST.__dict__['pars'])):
#            hyperparms.append(kernelFIRST.__dict__['pars'][k])
#        
#        #to save the 'old' kernel and gradient
#        first_kernel=new_kernel(kernelFIRST,hyperparms)
#        first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)  
#        
#        #update the hyperparms  x2=x1-step*grad1
#        new_hyperparms = [step*x for x in first_calc] #gives step*grad
#        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
#             
#        #new kernel with hyperparams updated    
#        second_kernel=new_kernel(kernelFIRST,new_hyperparms)
#        second_calc=sign_gradlike(second_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#    
#        #lets see if we are going the right direction
#        check_sign=[] #to check if we overshot the optimal value           
#        for i in range  (len(second_calc)):
#            check_sign.append(first_calc[i]*second_calc[i])
#        check_it=all(check_sign>0 for check_sign in check_sign)
#        #print check_it
#        if check_it is True: #everything is ok and things can continue                    
#            step=1.2*step #new bigger step to speed up the convergence            
#        else: #we passed the optimal value and need to go back
#            step=0.5*step #new smaller step to redo the calculations
#  
#    it=+1
#
#    grad_condition=0.001
#    while it<iterations and step>scipystep and minimum_grad>grad_condition:
#        #It's recommender to reset step*grad some times due to rouding errors                     
#        if it%(len(hyperparms)+1)!=0:  
#            print 'if'
#            print first_calc,second_calc
#            #this calc will give |deltaF1|**2
#            calc_aux1=[x**2 for x in first_calc]
#            calc_aux2=sum(calc_aux1)
#            #this calc will gives |deltaF2|**2
#            calc_aux3=[x**2 for x in second_calc]
#            calc_aux4=sum(calc_aux3)            
#            #this will give deltas = |deltaF2|**2/|deltaF1|**2
#            deltas=calc_aux4/calc_aux2 
#            
#            #new_hyperparms will be deltas*deltaF1
#            new_hyperparms = [x*deltas for x in first_calc]
#            #then it will be the sum of -deltaF2 with deltas*S1
#            #second_calcAUX=opt_gradlike(second_kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#            new_hyperparms = [-x+y for x,y in zip(second_calc, new_hyperparms)]             
#            print 'new hypwe',new_hyperparms
#            check_it = False
#            #the algorithm will only continue if we have the best step to give
#            while check_it is False and step>scipystep:
#                print 'second kerel', second_kernel
#                print 'second calc', second_calc
#                #new_hyperparms now gives step*S2
#                final_hyperparms = [x*step for x in new_hyperparms]
#                #new_hyperparms will finally give X3=X2 + lambda*S2
#                final_hyperparms = [x+y for x,y in zip(hyperparms, final_hyperparms)]                     
#                #new kernel with hyperparams updated    
#                third_kernel=new_kernel(kernelFIRST,final_hyperparms)
#                third_calc=sign_gradlike(third_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#                print 'third kernel',third_kernel
#                print 'third calc', third_calc
#
#                #lets see if we are going the right direction
#                check_sign=[] #to check if we overshot the optimal value           
#                for i in range  (len(second_calc)):
#                    check_sign.append(second_calc[i]*third_calc[i])
#                    check_it=all(check_sign>0 for check_sign in check_sign)
#                    if check_it is True: #everything is ok and things can continue                    
#                        print 'where if'                        
#                        step=1.2*step #new bigger step to speed up the convergence            
#                        
#                        #the second_kernel becomes the first_kernel
#                        hyperparms=[] #values of the hyperparameters 
#                        for k in range(len(second_kernel.__dict__['pars'])):
#                            hyperparms.append(second_kernel.__dict__['pars'][k])
#                        first_kernel=new_kernel(second_kernel,hyperparms)
#                        first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#
#                        #the third_kernel becomes the second_kernel
#                        hyperparms=[] #values of the hyperparameters 
#                        for k in range(len(third_kernel.__dict__['pars'])):
#                            hyperparms.append(third_kernel.__dict__['pars'][k])                        
#                        second_kernel=new_kernel(third_kernel,hyperparms)
#                        second_calc=sign_gradlike(third_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#                        hyperparms=[] #values of the hyperparameters 
#                        for k in range(len(second_kernel.__dict__['pars'])):
#                            hyperparms.append(second_kernel.__dict__['pars'][k])
#                        print first_kernel,second_kernel
#                    else: #we passed the optimal value and need to go back
#                        print 'else step'                       
#                        step=0.5*step #new smaller step to redo the calculations
#                print step
#            it+=1
#            #test of a stoping criteria
#            difference=[]
#            for i  in range(len(first_calc)):
#                difference.insert(0,abs(second_calc[i]))              
#                minimum_difference=np.min(difference)
#            minimum_grad=minimum_difference               
#
#        else:  
#            print 'else'
#            print first_kernel
#
#            check_it=False
#            #the algorithm will only continue if we have the best step to give
#            while check_it is False and step>scipystep: #and iterac <10:
#                print 'we got here'
#                hyperparms=[] #initial values of the hyperparameters 
#                for k in range(len(second_kernel.__dict__['pars'])):
#                    hyperparms.append(second_kernel.__dict__['pars'][k])
#                #print hyperparms
#                new_hyperparms = [x*step for x in second_calc]
#                new_hyperparms = [x+y for x,y in zip(hyperparms, new_hyperparms)]
#
#                #new kernel with hyperparams updated
#                first_kernel=second_kernel
#                first_calc=sign_gradlike(first_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#                second_kernel=new_kernel(kernelFIRST,new_hyperparms)
#                second_calc=sign_gradlike(second_kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
#                
#
#                #lets see if we are going the right direction
#                check_sign=[] #to check if we overshot the optimal value           
#                for i in range  (len(second_calc)):
#                    check_sign.append(first_calc[i]*second_calc[i])
#                check_it=all(check_sign>0 for check_sign in check_sign)
#                if check_it is True: #everything is ok and things can continue                    
#                    step=1.2*step #new bigger step to speed up the convergence
#                    check_it=True
#                else: #we passed the optimal value and need to go back
#                    step=0.5*step #new smaller step to redo the calculations
#                    check_it=False
#            it+=1
#            #test of a stoping criteria
#            difference=[]
#            for i  in range(len(first_calc)):
#                difference.insert(0,abs(second_calc[i]))              
#                minimum_difference=np.min(difference)
#            minimum_grad=minimum_difference
#    
#    #final likelihood and kernel
#    final_log=opt_likelihood(second_kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST)    
#    print 'total iterations ->', it
#    print 'final log likelihood ->',  final_log
#    print 'final kernel ->', second_kernel 