# Gedi 	<A href="https://www.quantifiedcode.com/app/project/6540466252654140acb80a5cdc12b8b0"><img src="https://www.quantifiedcode.com/api/v1/project/6540466252654140acb80a5cdc12b8b0/badge.svg" alt="Code issues"/></A>

Do or do not, there is no try in the use of Gaussian processes to model real data, test the limits of this approach, and find the best way to analyze radial velocities measurements of stars.
 
|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 
 How to install?
 The easy way is using pip: $ pip install Gedi
 
 What is the current version?
 Current version is Gedi 0.1.7 as of 30/04/2017. 

 What other packages are needed to work?
 It's necessary to have numpy, scipy and matplotlib.
 
|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

 List of available scripts on Github:
 
 kernel.py -> script with all the kernels and its derivatives.
 
 kernel_likelihood.py -> script where the  calculation of the log-likelihood and gradient of the kernels is made.

 kernel_optimization.py -> script where the kernel's optimizations is made

 kernel_mcmc.py -> a very simple script of a mcmc to optimize the kernels, it's not very efficient but it's a first step to create a good one.

 RV_function.py -> script to simulate rv signals.

 Tests.py -> simples tests to see how things work.

|▒▓▒▒◙▒▓▒▓▒▓||░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


