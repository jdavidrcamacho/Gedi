# -*- coding: utf-8 -*-
from distutils.core import setup
setup(
  name = 'Gedi',
  packages = ['Gedi'], # this must be the same as the name above
  version = '0.1.7',
  description = 'Package to analyze radial velocity measurements using Gaussian processes, still under development so be carefull using it ',
  author = 'Joao Camacho',
  author_email = 'joao.camacho@astro.up.pt',
  url = 'https://github.com/jdavidrcamacho/Gedi', # use the URL to the github repo
  download_url = 'https://github.com/jdavidrcamacho/Gedi/archive/0.1.7.tar.gz', 
  keywords = ['Gaussian', 'process','radial','velocity','exoplanet'], # arbitrary keywords
  classifiers = [
      'License :: OSI Approved :: MIT License'  
  ],
)
