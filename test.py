# -*- coding: utf-8 -*-

import gedi.gpCalc as calc
import gedi.gpKernel as kernels
import numpy as np
import matplotlib.pylab as pl

#data
x = 10 * np.sort(np.random.rand(101))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

#lets define a kernel
kernel0 = kernels.ExpSquared(10,1) * kernels.ExpSquared(5,0.3) + kernels.Exponential(1,1)

#lets calculate the log-likelihood
loglike = calc.likelihood(kernel0,x,y,yerr, kepler = False)
print(loglike)

#lets see the covariance kernel
k = calc.build_matrix(kernel0,y,yerr)
pl.imshow(k)

##### emcee example #####
import emcee
from matplotlib.ticker import MaxNLocator
from scipy import stats

#defining our priors
def logprob(p):
    global kernel
    if any([p[0] < np.log(1), p[0] > np.log(2), 
            p[1] < -10, p[1] > np.log(10),
            p[2] < np.log(4), p[2] > np.log(8), 
            p[3] < -10, p[3] > np.log(0.5)]):
        return -np.inf
    logprior=0.0
    # Update the kernel and compute the log marginal likelihood.
    kernel=calc.new_kernel(kernel,np.exp(p))
    new_likelihood=calc.likelihood(kernel,x,y,yerr)
    return logprior + new_likelihood

amplitude_prior=stats.uniform(1, 2-1)
lenghtscale_prior=stats.uniform(np.exp(-10), 10-np.exp(-10))
period_prior=stats.uniform(4, 8-4)
wn_prior=stats.uniform(np.exp(-10), 0.5-np.exp(-10))

def from_prior():
    return np.array([amplitude_prior.rvs(),lenghtscale_prior.rvs(),
                    period_prior.rvs(),wn_prior.rvs()])

#defining our kernel 
kernel=kernels.ExpSineSquared(amplitude_prior.rvs(),
                lenghtscale_prior.rvs(),period_prior.rvs()) +\
                kernels.WhiteNoise(wn_prior.rvs())

#preparing our MCMC
burns, runs= 2500, 5000

#set up the sampler.
nwalkers, ndim = 10, len(kernel.pars)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)

p0=[np.log(from_prior()) for i in range(nwalkers)]
assert not np.isinf(map(logprob, p0)).any()

p0, _, _ = sampler.run_mcmc(p0, burns)
sampler.run_mcmc(p0, runs)

fig, axes = pl.subplots(4, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4) #log
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$theta$")
axes[1].plot(np.exp(sampler.chain[:, :, 1]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$l$")
axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$P$")
axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4) #log
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$WN$")
axes[3].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)

burnin = 50
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

samples[:, 0] = np.exp(samples[:, 0])   #amplitude
samples[:, 1] = np.exp(samples[:, 1])   #lenght scale
samples[:, 2] = np.exp(samples[:, 2])   #period
samples[:, 3] = np.exp(samples[:, 3])   #white noise
theta_mcmc,l_mcmc,p_mcmc,wn_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print('theta = {0[0]} +{0[1]} -{0[2]}'.format(theta_mcmc))
print('l = {0[0]} +{0[1]} -{0[2]}'.format(l_mcmc))
print('period = {0[0]} +{0[1]} -{0[2]}'.format(p_mcmc))
print('white noise = {0[0]} +{0[1]} -{0[2]}'.format(wn_mcmc))
