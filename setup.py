# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:16:28 2019

@author: Asif Towheed
"""

from distutils.core import setup
from Cython.Build import cythonize
 
setup(
    ext_modules = cythonize("AD_Functions_22.pyx")
)
