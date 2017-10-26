# Gedi - introducing the gaussian jedi	

Do or do not, there is no try in the use of Gaussian processes to model real data, test the limits of this approach, and find the best way to analyze radial velocities measurements of stars.
 



|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 

 How to install?

 The easy way is using pip: 
```
$ pip install Gedi
``` 

 What is the current version?

 Current version is Gedi 0.2 as of 29/09/2017.


 What other packages are needed to work?

 It's necessary to have numpy, scipy and matplotlib.
 



|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

## Gedi examples

In this section we present two simple examples of how to work with *gedi*. In the first it will be presented a simple example on how to perform the optimization of the kernel with *scipy.optimize*. The second one will be another simple example but this time using the package *emcee*. With this two example we believe we are able to show the full potential of *gedi* and of the functions it has implemented.


### *scipy.optimize*

To use *gedi* together with *scipy.optimize* we first we of course to import all the necessary python packages. Besides the two mentioned packages we will also require *numpy*.

```
import numpy as np
import Gedi as gedi
import scipy.optimize as op
```

Having imported all the necessary packages we can now simulate some some sinusoidal data.

```
np.random.seed(1001)
x= 10 * np.sort(np.random.rand(30))
yerr= 0.5 * np.ones_like(x)
y= np.sin(x) + yerr * np.random.randn(len(x))
pl.plot(x,y,'*')
```

To have some consistency in the examples showed in this appendix we also define a seed to allow the reproduction of the results in the future.

With the generated data we can now choose a kernel to use with it and calculate the respective log marginal likelihood

```
#first kernel
kernel0= gedi.kernel.ExpSineSquared(2,2.5,5)
kernel_lk0= gedi.kernel_likelihood.likelihood(kernel0,x,y,yerr)
print('initial likelihood',kernel_lk0)

#second kernel
kernel= gedi.kernel.ExpSineSquared(2,2.5,5) + gedi.kernel.WhiteNoise(0.2)
kernel_lk= gedi.kernel_likelihood.likelihood(kernel,x,y,yerr)
print('initial likelihood',kernel_lk)
```

Since we are working with a sinusoid it is logical to use the *ES kernel* as it is a periodic kernel. From the first kernel we were able to obtain a log marginal likelihood of around -51.06, while with the second we obtained around -47.97. This clearly tell us that the second kernel is a better choice to work with, what comes with not much surprise, as the generated data contained noise.

As such we will continue our analysis using the second kernel and now define the log marginal likelihood and the gradients that *scipy.optimize* will use


```
#Log marginal likelihood
def likelihood_gedi(p):
    global kernel
    # Update the kernel parameters and compute the likelihood.
    kernel= gedi.kernel_optimization.new_kernel(kernel,np.exp(p))
    ll = gedi.kernel_likelihood.likelihood(kernel,x,y,yerr)
    return -ll if np.isfinite(ll) else 1e25

#Gradients
def gradients_gedi(p):
    global kernel
    # Update the kernel parameters and compute the likelihood.
    kernel= gedi.kernel_optimization.new_kernel(kernel,np.exp(p))
    return -np.array(gedi.kernel_likelihood.gradient_likelihood(kernel,x,y,yerr))
```

With this two simple functions we are now ready to use \textit{scipy.optimize} and find our optimized kernel

```
#lets run the optimization
p0_gedi = np.log(kernel.pars)
results_gedi = op.minimize(likelihood_gedi, p0_gedi, jac=gradients_gedi)
kernel= gedi.kernel_optimization.new_kernel(kernel,np.exp(results_gedi.x))

print('Final kernel',kernel)
print('Final likelihood =',gedi.kernel_likelihood.likelihood(kernel,x,y,yerr))
```

This allow us to obtain as a final result

```
('Final kernel', ExpSineSquared(2.72896536025, 1.10953213369, 16.5826072023)
+ WhiteNoise(0.356778865898))
('Final likelihood', -34.288489695783142)
```

The final log marginal likelihood show us that the final kernel is indeed a better kernel than the one used in the beginning, and *scipy.optimize* was successful in finding a better solution to the one we had.

### emcee

We are now going to use *gedi* in conjunction with *emcee* to find the best values of our kernel's hyperparameters. We obviously begin by importing all necessary packages

```
import Gedi as gedi
import emcee
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
```

Besides *gedi* and *emcee* we will also need to use *numpy*, *scipy* and *matplotlib*, to generate our data and plot the necessary graphics. To start this second example we can generate same sinusoidal data with and respective error with

```
np.random.seed(1001)
x= 10 * np.sort(np.random.rand(30))
yerr= 0.2 * np.ones_like(x)
y= np.sin(x) + yerr * np.random.randn(len(x))
```

With the help of *matplotlib* we can take a look on our generated data.

```
pl.plot(x,y,'*')
pl.xlabel('x')
pl.ylabel('y')
```

![](https://i.imgur.com/rbTVvQM.png)

This allow us to see that the amplitude of our data is around 1 unit and the period around 6 units, which will be useful to set our priors. Having prepared the data that we will be working with we can now prepare our MCMC and start by defining our priors and the log marginal likelihood we will use

```
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
    kernel=gedi.kernel_optimization.new_kernel(kernel,np.exp(p))
    new_likelihood=gedi.kernel_likelihood.likelihood(kernel,x,y,yerr)
    return logprior + new_likelihood

amplitude_prior=stats.uniform(1, 2-1)
lenghtscale_prior=stats.uniform(np.exp(-10), 10-np.exp(-10))
period_prior=stats.uniform(4, 8-4)
wn_prior=stats.uniform(np.exp(-10), 0.5-np.exp(-10))

def from_prior():
    return np.array([amplitude_prior.rvs(),lenghtscale_prior.rvs(),
                    period_prior.rvs(),wn_prior.rvs()])

#defining our kernel 
kernel=gedi.kernel.ExpSineSquared(amplitude_prior.rvs(),
                lenghtscale_prior.rvs(),period_prior.rvs()) +\
                gedi.kernel.WhiteNoise(wn_prior.rvs())

#preparing our MCMC
burns, runs= 2500, 5000

#set up the sampler.
nwalkers, ndim = 10, len(kernel.pars)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)

p0=[np.log(from_prior()) for i in range(nwalkers)]
assert not np.isinf(map(lnprob, p0)).any()

p0, _, _ = sampler.run_mcmc(p0, burns)
sampler.run_mcmc(p0, runs)
```

Once again we use the sum of an *ES kernel* with a *WN kernel* to fit to our data, since our data comes from a sinusoidal model. With our MCMC complete we can now plot the results to check visually if we had convergence in our hyperparameters.

```
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
```

![](https://i.imgur.com/k5GDVey.png)

Using 5000 steps as our burn-in and 5000 step to use in our MCMC, we can see that there seems to be a convergence, for example, on the hyperparameter that corresponds to the period and the white noise of the kernel, although the same seemed to not be obtained to the amplitude and the length-scale, as this is just an example of how to use *gedi*, we will ignore it and continue on our analysis.

To obtain our final solution we can compute the quantiles and median that will be used

```
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
```

In the end of this we are able to obtain the values

```
theta = 1.41668244894 +0.373790095626 -0.288328791884
l = 2.46889155829 +0.952416345357 -0.735937463898
period = 6.42794862032 +0.096144793889 -0.0969172051794
white noise = 0.00225458063694 +0.0358745326978 -0.00209754174694
```

Which comparing with the value obtained in the optimization with *scipy.optimize*, using a MCMC has a greater advantage. Not only we were able to obtain a value for our hyperparameters, we were able to obtain an error interval for each one of it. Since the optimization of the hyperparameters with gradient based algorithms is not a convex problem, it is necessary to be careful about our initial values, in order to not reach a bad local minima.

The result obtained with *scipy.optimize* give us a period of around 16.58 units, while just by looking at the graph of the data analyzed, we should have a periodicity around 6 units. Unlike *scipy.optimize*, the MCMC clearly reach this value, showing us that the problem of bad local minima does not occur with the use of an MCMC.




|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


