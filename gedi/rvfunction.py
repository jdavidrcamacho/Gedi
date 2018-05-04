# -*- coding: utf-8 -*-
import  numpy as _np

##### RV functions #####
def RV_circular(P=365, K=0.1, T=0, gamma=0, t=None):
    """
        RV_circular() simulates the radial velocity signal of a planet in a 
    circular orbit around a star.
        The algorithm needs improvements since it is inefficient.

        Parameters:
    P = period in days
    K =  semi-amplitude of the signal
    T = velocity at zero phase
    gamma = average velocity of the star
    t  = time
    space = We want an observation every time/space days

        Returns:
    t = time
    RV = rv signal generated    
    """ 
    if t is  None:
    	print('Time needed')

    RV = [K*_np.sin(2*_np.pi*x/P - T) + gamma for x in t]
    RV = [x for x in RV] #m/s 
    return t, RV


def RV_kepler(P=365, e=0, K=.1, T=0, gamma=0, w=_np.pi, t=None):
    """
        RV_kepler() simulates the radial velocity signal of a planet in a 
    keplerian orbit around a star.

        Parameters:
    P = period in days
    e = eccentricity
    K = RV amplitude
    gamma = constant system RV
    T = zero phase
    w = longitude of the periastron
    t = time

        Returns:
    t = time
    RV = rv signal generated
    """
    if t is  None:
    	print('Time needed')

    #mean anomaly
    mean_anom = [2*_np.pi*(x1-T)/P  for x1 in t]

    #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0 = [x + e*_np.sin(x)  + 0.5*(e**2)*_np.sin(2*x) for x in mean_anom]
    #mean anomaly -> M0=E0 - e*sin(E0)
    M0 = [x - e*_np.sin(x) for x in E0]

    i = 0
    while i < 100:
        #[x + y for x, y in zip(first, second)]
        calc_aux = [x2-y for x2,y in zip(mean_anom, M0)]    
        E1 = [x3 + y/(1-e*_np.cos(x3)) for x3,y in zip(E0, calc_aux)]
        M1 = [x4 - e*_np.sin(x4) for x4 in E0]   
        i += 1
        E0 = E1
        M0 = M1

    nu = [2*_np.arctan(_np.sqrt((1+e)/(1-e))*_np.tan(x5/2)) for x5 in E0]
    RV = [ gamma + K*(e*_np.cos(w)+_np.cos(w+x6)) for x6 in nu]
    RV = [x for x in RV] #m/s 
    return t, RV
