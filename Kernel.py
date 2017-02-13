# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
    
##### KERNELS 
class Kernel(object):
    def __init__(self, *args):
        self.pars = np.array(args) # put all Kernel arguments in an array pars

    def __call__(self, r):
        raise NotImplementedError
        #return self.k1(x1, x2, i, j) * self.k2(x1, x2, i, j)

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Product(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)

    def __repr__(self):
        """ Representation of each Kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    def change_params(self):
        return Kernel(self)
                             
class _operator(Kernel):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator): #sum of kernels
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)

    def parSize(self):
        return self.pars.size   


class Product(_operator): #multiplication of kernels
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, r):
        return self.k1(r) * self.k2(r)

    def parSize(self):
        return self.pars.size
        
    def prod_kernel(self, r):
        return self.k1(r), self.k2(r)

class ExpSquared(Kernel):
    def __init__(self, ES_theta, ES_l):
        super(ExpSquared, self).__init__(ES_theta, ES_l)
        # because we are "overwriting" the function __init__
        # we use this weird super function
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l
        
    def __call__(self, r):
        f1 = self.ES_theta**2   #theta**2
        f2 = self.ES_l**2       #l**2
        f3 = (r)**2             #(x1-x2)**2
        return f1 * np.exp(-0.5* f3/f2)

    def dES_dtheta(self, r):        
        f1=self.ES_theta**2    #theta        
        f2=self.ES_l**2     #l**2
        f3=(r)**2       #(x1-x2)**2
        return  2*f1*np.exp(-0.5*f3/f2)

    def dES_dl(self, r):
        f1=self.ES_theta**2     #theta**2
        f2=self.ES_l            #l
        f3=(r)**2               #(x1-x2)**2
        f4=self.ES_l**3         #l**3
        return f1*(f3/f4)*np.exp(-0.5*f3/f2**2) *f2#mult por l
   
     
class ExpSineSquared(Kernel):
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(ExpSineSquared, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
    
    def __call__(self, r):
        f1 = self.ESS_theta**2
        f2 = self.ESS_l**2
        f3 = np.abs(r)
        f4 = self.ESS_P
        return f1*np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2))   
      
    def dESS_dtheta(self,r):
        f1 = self.ESS_theta**2  #theta**2
        f2 = self.ESS_l**2      #l**2 
        f3 = np.pi/self.ESS_P   #pi/P
        f4 = np.abs(r)
        return 2*f1*np.exp(-(2.0/f2)*np.sin(f3*f4)**2)  
    
    def dESS_dl(self,r):
        f1=self.ESS_theta**2
        f2=self.ESS_l**3
        f3=np.pi/self.ESS_P
        f4=np.abs(r)
        f5=self.ESS_l**2
        f6=self.ESS_l
        return (4*f1/f2)*(np.sin(f3*f4)**2)*np.exp((-2./f5)*np.sin(f3*f4)**2) \
                *f6#mult por l 
        
    def dESS_dP(self,r):
        f1=self.ESS_theta**2    #theta**2
        f2=self.ESS_l**2        #l**2
        f3=np.pi/self.ESS_P     #pi/P      
        f4=self.ESS_P           #P
        f5=np.abs(r)            #x1-x2 ou x2-x1
        return f1*(4./f2)*f3*f5*np.cos(f3*f5)*np.sin(f3*f5) \
                *np.exp((-2.0/f2)*np.sin(f3*f5)**2) 

      
class RatQuadratic(Kernel):
    def __init__(self, RQ_theta, RQ_alpha, RQ_l):
        super(RatQuadratic, self).__init__(RQ_theta, RQ_alpha, RQ_l)
        self.RQ_theta = RQ_theta
        self.RQ_alpha = RQ_alpha
        self.RQ_l = RQ_l
    
    def __call__(self, r):
        f1 = self.RQ_theta**2
        f11 = self.RQ_theta
        f2 = self.RQ_l**2
        f3 = (r)**2
        f4 = self.RQ_alpha
        return f1*(1+(0.5*f3/(f4*f2)))**(-f4)
            
    def dRQ_dtheta(self,r):
        f1=self.RQ_theta**2     #theta**2
        f2=(x1-x2)**2           #(x1-x2)**2
        f3=self.RQ_alpha        #alpha
        f4=self.RQ_l**2         #l**2
        return 2*f1*(1.0 + f2/(2.0*f3*f4))**(-f3)
 
    def dRQ_dl(self,r):
        f1=self.RQ_theta**2     #theta**2
        f2=(r)**2               #(x1-x2)**2     
        f3=self.RQ_alpha        #alpha
        f4=self.RQ_l**2         #l**2
        #f5=self.RQ_l**3        #l**3
        return (f1*f2/f4)*(1.0 + f2/(2.0*f3*f4))**(-1.0-f3)
        
    def dRQ_dalpha(self,r):
        f1=self.RQ_theta**2     #theta**2
        f2=(r)**2               #(x1-x2)**2
        f3=self.RQ_alpha        #alpha
        f4=self.RQ_l**2         #l**2
        func0=1.0 + f2/(2.0*f3*f4)
        func1=f2/(2.0*f3*f4*func0)
        return f1*(func1-np.log(func0))*func0**(-f3) *f3       
    
        
class WhiteNoise(Kernel):                             
    def __init__(self,WN_theta):                     
        super(WhiteNoise,self).__init__(WN_theta)     
        self.WN_theta=WN_theta                        
                                                      
    def __call__(self, r):
        f1=self.WN_theta**2
        f2=np.diag(np.ones_like(r))
        return f1*f2 

    def dWN_dtheta(self,r):
        f1=self.WN_theta**2  #theta
        f2=np.diag(np.ones_like(r))
        return 2*f1*f2      
                

class Exponential(Kernel): #Matern 1/2 = Exponential
    def __init__(self,Exp_theta,Exp_l):
        super(Exponential,self).__init__(Exp_theta,Exp_l)
        self.Exp_theta=Exp_theta        
        self.Exp_l=Exp_l

    def __call__(self, r):
        f1=r
        f2=self.Exp_l
        f3=self.Exp_theta**2
        return f3*np.exp(-f1/f2)

    def dExp_dtheta(self,r):
        f1=self.Exp_theta**2  #theta
        f2=(r)                #(x1-x2)**2
        f3=self.Exp_l         #l
        return 2*f1*np.exp(-f2/f3)        
        
    def dExp_dl(self,r):
        f1=self.Exp_theta**2  #theta**2
        f2=(r)                #(x1-x2)
        f3=self.Exp_l         #l
        f4=self.Exp_l**2      #l**2
        return (f1*f2/f3)*np.exp(-f2/f3)
    

class Matern_32(Kernel): #Matern 3/2
    def __init__(self,M32_theta,M32_l):
        super(Matern_32,self).__init__(M32_theta,M32_l)
        self.M32_theta=M32_theta   
        self.M32_l=M32_l  
        
    def __call__(self, r):
        f1=np.sqrt(3.0)*(r)
        f2=self.M32_l
        f3=self.M32_theta**2
        return f3*(1.0 + f1/f2)*np.exp(-f1/f2)

    def dM32_dtheta(self,r):
        f1=self.M32_theta**2  #theta
        f2=r                  #(x1-x2)**2
        f3=self.M32_l         #l
        return 2*f1*(1+np.sqrt(3)/f3)*np.exp(-np.sqrt(3)*f2/f3)        
        
    def dM32_dl(self,r):
        f1=self.M32_theta**2    #theta**2
        f2=r                    #(x1-x2)
        f3=self.M32_l           #l
        f4=self.M32_l**2        #l**2
        return f1*np.sqrt(3)*f2*(1+np.sqrt(3)/f3)*np.exp(-np.sqrt(3)*f2/f3)/f4 \
                - f1*np.sqrt(3)*np.exp(-np.sqrt(3)*f2/f3)/f4 

        
class Matern_52(Kernel): #Matern 5/2
    def __init__(self,M52_l):
        super(Matern_52,self).__init__(M52_theta,M52_l)
        self.M52_theta=M52_theta        
        self.M52_l=M52_l

    def __call__(self, r):
        f1=np.sqrt(5.0)*r
        f2=(r)**2        
        f3=self.M52_l
        f4=self.M52_l**2
        f5=self.M52_theta**2
        return f5*(1.0 + f1/f3 + (5.0*f4)/(3.0*f4))*np.exp(-f1/f3)
    
    def dM52_dtheta(self,r):
        f1=self.M52_theta**2      #theta
        f2=self.M52_l             #l
        f3=self.M52_l**2          #l**2
        f4=r                      #(x1-x2)
        f5=r**2                   #(x1-x2)**2
        return 2*f1*(1 + np.sqrt(5)*f4/f2 + (5.*f5)/(3*f3)) \
                *np.exp(-np.sqrt(5)*f4/f2)
         
    def dM52_dl(self,r):
        f1=self.M52_theta**2  #theta
        f2=self.M52_l         #l
        f3=self.M52_l**2      #l**2
        f4=r                  #(x1-x2)
        f5=r**2               #(x1-x2)**2
        return (f1/f3)*np.sqrt(5)*(1+np.sqrt(5)*f4/f2+(5.*f5)/(3*f3))*f4*np.exp(-np.sqrt(5)*f4/f2) \
                + f1*(-np.sqrt(5)*f4/f3-(10.*f5)/(3*f2*f3))*np.exp(-np.sqrt(5)*f4/f2)
                
                
### This kernel is equal to george's ExpSine2Kernel
class  ExpSineGeorge(Kernel):
    def __init__(self,gamma,P):
        super(ExpSineGeorge,self).__init__(gamma,P)
        self.gamma=gamma
        self.P=P
        
    def __call__(self,r):
        f1=self.gamma
        f2=self.P
        f3=r
        return np.exp(-f1 *  np.sin(np.pi*f3/f2)**2)
        
    def dE_dGamma(self,r):
        f1 = self.gamma
        f2 = self.P
        f3 = r
        f4 = -np.sin(np.pi*f3/f2)**2
        f5 = np.exp(-f1*np.sin(np.pi*f3/f2)**2)  
        return f4*f5*f1
        
    def dE_dP(self,r):
        f1 = self.gamma
        f2 = self.P
        f3 = r
        f4 = np.sin(np.pi*f3/f2)
        f5 = np.cos(np.pi*f3/f2)
        f6 = np.exp(-f1 *  np.sin(np.pi*f3/f2)**2)
        return 2*f1*(np.pi*f3/f2)*f4*f5*(f6/f2)*f2

