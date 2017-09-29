# -*- coding: utf-8 -*-
from distutils.core import setup
setup(
  name = 'Gedi',
  packages = ['Gedi'], # this must be the same as the name above
  version = '0.2',
  description = 'Package to analyze radial velocity measurements using Gaussian processes made for a MSc Thesis',
  author = 'Joao Camacho',
  author_email = 'joao.camacho@astro.up.pt',
  url = 'https://github.com/jdavidrcamacho/Gedi', # use the URL to the github repo
  download_url = 'https://github.com/jdavidrcamacho/Gedi/archive/0.2.tar.gz', 
  keywords = ['Gaussian', 'process','radial','velocity','exoplanet'], # arbitrary keywords
  classifiers = [
      'License :: OSI Approved :: MIT License'  
  ],
)
