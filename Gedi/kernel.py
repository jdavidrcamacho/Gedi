# -*- coding: utf-8 -*-
import numpy as np


class Kernel(object):
    """ 
        Definition the kernels and its properties, 
    that includes sum and multiplication of kernels. 
    """
    def __init__(self, *args):
        """ puts all Kernel arguments in an array pars """
        self.pars = np.array(args)

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

                            
class _operator(Kernel):
    """ To allow operations between two kernels """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ To allow the sum of kernels """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Product(_operator):
    """ To allow the multiplycation of kernels """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, r):
        return self.k1(r) * self.k2(r)


class ExpSquared(Kernel):
    """
        Definition of the exponential squared kernel and its derivatives,
    it is also know as radial basis function (RBF kernel).
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    ES_theta = amplitude of the kernel
    ES_l = characteristic lenght scale  to define how smooth the kernel is   
    """
    def __init__(self, ES_theta, ES_l):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(ExpSquared, self).__init__(ES_theta, ES_l)
        self.ES_theta = ES_theta
        self.ES_l = ES_l
        
    def __call__(self, r):
        f1 = self.ES_theta**2   
        f2 = self.ES_l**2       
        f3 = (r)**2             
        return f1 * np.exp(-0.5* f3/f2)
    
    def des_dtheta(self, r):
        """ Log-derivative in order to theta """        
        f1=self.ES_theta**2      
        f2=self.ES_l**2
        f3=(r)**2
        return  2*f1*np.exp(-0.5*f3/f2)
    
    def des_dl(self, r):
        """ Log-derivative in order to l """
        f1=self.ES_theta**2
        f2=self.ES_l
        f3=(r)**2
        f4=self.ES_l**3    
        return f1*(f3/f4)*np.exp(-0.5*f3/f2**2) *f2
   
   
class ExpSineSquared(Kernel):
    """
        Definition of the exponential sine squared kernel and its derivatives,
    it is also know as periodic kernel.
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    ESS_theta = amplitude of the kernel
    ESS_l = characteristic lenght scale  to define how smooth the kernel is   
    ESS_P = periodic repetitions of the kernel
    """
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
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
        
    def dess_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1 = self.ESS_theta**2
        f2 = self.ESS_l**2
        f3 = np.pi/self.ESS_P
        f4 = np.abs(r)
        return 2*f1*np.exp(-(2.0/f2)*np.sin(f3*f4)**2)  
    
    def dess_dl(self,r):
        """ Log-derivative in order to l """
        f1=self.ESS_theta**2
        f2=self.ESS_l**3
        f3=np.pi/self.ESS_P
        f4=np.abs(r)
        f5=self.ESS_l**2
        f6=self.ESS_l
        return (4*f1/f2)*(np.sin(f3*f4)**2)*np.exp((-2./f5)*np.sin(f3*f4)**2) \
                *f6
    
 
    def dess_dp(self,r):
        """ Log-derivative in order to P """
        f1=self.ESS_theta**2
        f2=self.ESS_l**2
        f3=np.pi/self.ESS_P
        f5=np.abs(r)
        return f1*(4./f2)*f3*f5*np.cos(f3*f5)*np.sin(f3*f5) \
                *np.exp((-2.0/f2)*np.sin(f3*f5)**2) 


class QuasiPeriodic(Kernel):
    """
        Definition of the product between the exponential sine squared kernel 
    and the exponential squared kernel, also known as quasi periodic kernel.
        I define this kernel because it is widely used and makes things more
    efficient to run instead of multiplying two kernels and make GEDI to run.
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    QP_theta = amplitude of the kernel
    QP_l1 and QP_l2 = characteristic lenght scales to define how 
                        smooth the kernel is   
    QP_P = periodic repetitions of the kernel
    """
    def __init__(self, QP_theta, QP_l1, QP_l2, QP_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(QuasiPeriodic, self).__init__(QP_theta, QP_l1, QP_l2, QP_P)
        self.QP_theta = QP_theta
        self.QP_l1 = QP_l1
        self.QP_l2 = QP_l2
        self.QP_P = QP_P    

    def __call__(self, r):
        f1 = self.QP_theta**2
        f2 = self.QP_l1**2
        ff2= self.QP_l2**2
        f3 = np.abs(r)
        f4 = self.QP_P
        return f1*np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2)-(0.5*f3*f3/ff2))
                
    def dqp_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1 = self.QP_theta**2
        f2 = self.QP_l1**2
        ff2= self.QP_l2**2
        f3 = np.abs(r)
        f4 = self.QP_P
        return 2*f1*np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2)-(0.5*f3*f3/ff2))
                
    def dqp_dl1(self,r):
        """ Log-derivative in order to l1 """
        f1 = self.QP_theta**2
        f2 = self.QP_l1**2
        ff2= self.QP_l2**2
        f3 = np.abs(r)
        f4 = self.QP_P
        return 4*f1*((np.sin(np.pi*f3/f4))**2)/f2 \
                *np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2)-(0.5*f3*f3/ff2))

    def dqp_dl2(self,r):    
        """ Log-derivative in order to l2 """
        f1 = self.QP_theta**2
        f2 = self.QP_l1**2
        ff2= self.QP_l2**2
        f3 = np.abs(r)
        f4 = self.QP_P
        return f1*f3*f3/ff2 \
                *np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2)-(0.5*f3*f3/ff2))
        
    def dqp_dp(self,r):
        """ Log-derivative in order to P """
        f1 = self.QP_theta**2
        f2 = self.QP_l1**2
        ff2= self.QP_l2**2
        f3 = np.abs(r)
        f4 = self.QP_P
        return 4*np.pi*f1*np.cos(np.pi*f3/f4)*np.sin(np.pi*f3/f4)/(f2*f4) \
                *np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2)-(0.5*f3*f3/ff2))


class RatQuadratic(Kernel):
    """
        Definition of the rational quadratic kernel and its derivatives.
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    RQ_theta = amplitude of the kernel
    RQ_alpha = weight of large and small scale variations
    RQ_l = characteristic lenght scale to define how smooth the kernel is   
    """
    def __init__(self, RQ_theta, RQ_alpha, RQ_l):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(RatQuadratic, self).__init__(RQ_theta, RQ_alpha, RQ_l)
        self.RQ_theta = RQ_theta
        self.RQ_alpha = RQ_alpha
        self.RQ_l = RQ_l
    
    def __call__(self, r):
        f1 = self.RQ_theta**2
        f2 = self.RQ_l**2
        f3 = (r)**2
        f4 = self.RQ_alpha
        return f1*(1+(0.5*f3/(f4*f2)))**(-f4)
    
    def drq_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1=self.RQ_theta**2
        f2=(r)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        return 2*f1*(1.0 + f2/(2.0*f3*f4))**(-f3)

    def drq_dl(self,r):
        """ Log-derivatives in order to l """
        f1=self.RQ_theta**2
        f2=(r)**2    
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        return (f1*f2/f4)*(1.0 + f2/(2.0*f3*f4))**(-1.0-f3)
        
    def drq_dalpha(self,r):
        """ Log-derivative in order to alpha """
        f1=self.RQ_theta**2
        f2=(r)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        func0=1.0 + f2/(2.0*f3*f4)
        func1=f2/(2.0*f3*f4*func0)
        return f1*(func1-np.log(func0))*func0**(-f3) *f3       
    

class WhiteNoise(Kernel):                             
    """
        Definition of the white noise kernel and its derivatives.
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    WN_theta = amplitude of the kernel
    """ 
    def __init__(self,WN_theta):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """                    
        super(WhiteNoise,self).__init__(WN_theta)     
        self.WN_theta=WN_theta                        
                                                      
    def __call__(self, r):
        f1=self.WN_theta**2
        f2=np.diag(np.diag(np.ones_like(r)))
        return f1*f2 

    def dwn_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1=self.WN_theta**2
        f2=np.diag(np.diag(np.ones_like(r)))
        return 2*f1*f2      

             
class Exponential(Kernel):
    """
        Definition of the exponential kernel and its derivatives, this kernel
    arise when setting v=1/2 in the matern family of kernels
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    EXP_theta = amplitude of the kernel
    EXP_l = characteristic lenght scale to define how smooth the kernel is  
    """  
    def __init__(self,Exp_theta,Exp_l):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(Exponential,self).__init__(Exp_theta,Exp_l)
        self.Exp_theta=Exp_theta        
        self.Exp_l=Exp_l

    def __call__(self, r):
        f1=np.abs(r)
        f2=self.Exp_l
        f3=self.Exp_theta**2
        return f3*np.exp(-f1/f2)

    def dexp_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1=np.abs(r)
        f2=self.Exp_l
        f3=self.Exp_theta**2
        return 2*f3*np.exp(-f1/f2)      
    
    def dexp_dl(self,r):
        """ Log-derivative in order to l """
        f1=self.Exp_theta**2  
        f2=np.abs(r)          
        f3=self.Exp_l         
        return (f1*f2/f3)*np.exp(-f2/f3)


class Matern32(Kernel):
    """
        Definition of the Matern 3/2 kernel and its derivatives, this kernel
    arise when setting v=3/2 in the matern family of kernels
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    M32_theta = amplitude of the kernel
    M32_l = characteristic lenght scale to define how smooth the kernel is  
    """ 
    def __init__(self,M32_theta,M32_l):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(Matern32,self).__init__(M32_theta,M32_l)
        self.M32_theta=M32_theta   
        self.M32_l=M32_l  
        
    def __call__(self, r): 
        f1=np.sqrt(3.0)*np.abs(r) 
        f2=self.M32_l
        f3=self.M32_theta**2
        return f3*(1.0 + f1/f2)*np.exp(-f1/f2)

    def dm32_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1=np.sqrt(3.0)*np.abs(r) 
        f2=self.M32_l
        f3=self.M32_theta**2
        return 2*f3*(1.0 + f1/f2)*np.exp(-f1/f2)   
    
    def dm32_dl(self,r):
        """ Log-derivative in order to l """
        f1=self.M32_theta**2        
        f2=np.sqrt(3.0)*np.abs(r)   
        f3=self.M32_l               
        f4=self.M32_l**2            
        return f3*f1*(f2/f4)*(1+f2/f3)*np.exp(-f2/f3) \
                - f3*f1*(f2/f4)*np.exp(-f2/f3)


class Matern52(Kernel):
    """
        Definition of the Matern 5/2 kernel and its derivatives, this kernel
    arise when setting v=5/2 in the matern family of kernels
    
        Important
    The derivative its in respect to log(parameter)
    
        Parameters
    M52_theta = amplitude of the kernel
    M52_l = characteristic lenght scale to define how smooth the kernel is  
    """ 
    def __init__(self,M52_theta,M52_l):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(Matern52,self).__init__(M52_theta,M52_l)
        self.M52_theta=M52_theta        
        self.M52_l=M52_l

    def __call__(self, r):
        f1=np.sqrt(5.0)*np.abs(r)
        f2=(np.abs(r))**2        
        f3=self.M52_l
        f4=self.M52_l**2
        f5=self.M52_theta**2
        return f5*(1.0 + f1/f3 + (5.0*f2)/(3.0*f4))*np.exp(-f1/f3)
    
    def dm52_dtheta(self,r):
        """ Log-derivative in order to theta """
        f1=self.M52_theta**2
        f2=self.M52_l
        f3=3*(self.M52_l)**2
        f4=np.sqrt(5)*np.abs(r)
        f5=5*np.abs(r)**2
        return 2*f1*(f5/f3 + f4/f2 +1)*np.exp(-f4/f2)
         
    def dm52_dl(self,r):
        """ Log-derivative in order to l """
        f1=self.M52_theta**2
        f2=self.M52_l
        f3=self.M52_l**2
        f4=np.abs(r)
        f5=np.abs(r)**2
        return 2*f1*((5*f2*f5 + np.sqrt(5**3)*f5*f4)/(3*f3*f2) \
                *np.exp(-np.sqrt(5)*f4/f2))                
                
##### END