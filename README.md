# Gedi

Do or do not, there is no try in the use of Gaussian processes to model real data, test the limits of this approach, and find the best way to analyze radial velocities measurements of stars.
 
|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 
 How to install?
 The easy way is using pip: $ pip install Gedi

 What other packages are needed to work?
 It's necessary to have numpy, scipy, matplotlib, inspect and time.
 The last one is only needed because I forgot to delete "from time import time" from one of the scripts before releasing Gedi 0.1
 
|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

 List of available scripts:
 
 kernel.py -> script with all the kernels and its derivatives
 
 kernel_likelihood.py -> script where the  calculation of the log-likelihood and gradient of the kernels is made

 kernel_optimization.py -> script where the kernel's optimizations is made, it's not in its final form (damn Android 18 is hidden somewhere) 

 kernel_mcmc.py -> a very simple script of a mcmc to optimize the kernels, it's not very efficient but it's a first step to create a good one

 Tests.py -> simples tests to see how things work

|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
