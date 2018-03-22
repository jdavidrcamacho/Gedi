# -*- coding: utf-8 -*-

from setuptools import setup

setup(	name = 'gedi',
	packages = ['gedi'], 
	version = '0.3',
	description = 'Package to analyze radial velocity measurements using Gaussian processes made for a MSc Thesis',
	author = 'Joao Camacho',
	author_email = 'joao.camacho@astro.up.pt',
	license='MIT',
	url = 'https://github.com/jdavidrcamacho/Gedi', 
	keywords = ['Gaussian', 'process','radial','velocity','exoplanet'],
	classifiers = ['License :: OSI Approved :: MIT License'],
	install_requires=[
        'numpy',
        'scipy',
        'matplotlib>=1.5.3',
        'emcee'
      ],
     )
