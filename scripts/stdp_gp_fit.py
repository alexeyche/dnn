#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:04:19 2015

@author: alexeyche
"""

import numpy as np    

from lib.bo_validation import run_search
from lib.bo_validation import run_validation
from lib.bo_validation import get_validation_params
from lib.bo_validation import plot_validation
from lib.env import runs_dir, cases_dir
from os.path import join as pj

data_file = pj(cases_dir, "lrule", "stdp_mc.ssv")
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = data[:,-1]


nfold = 2
kernels_to_search = ["kRQISO", "kSEARD", "kMaternISO1", "kMaternISO3", "kMaternISO5", "kMaternARD1", "kMaternARD3", "kMaternARD5"]

#kernels = [
#    "kMaternARD1", "kMaternISO1",
#]
#comp_kernels = ["kSum", "kProd"]
#


results = {}

for m in ["sGaussianProcessML", "sStudentTProcessJef", "sStudentTProcessNIG", "sGaussianProcess"]:
    print "Searching for {}".format(m)    
    res = run_search(X, Y, kernels_to_search, nfold, params = {"surr_name": m}, generate_plots=False)
    results[m] = res

for m, kres in results.iteritems():
    print m
    for k, score in kres:
        print "\t{} {}".format(k, score)