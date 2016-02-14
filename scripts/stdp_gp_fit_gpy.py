#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:04:19 2015

@author: alexeyche
"""

import numpy as np    
import GPy

from lib.bo_validation import run_search
from lib.bo_validation import create_model
from lib.bo_validation import run_validation
from lib.bo_validation import get_validation_params
from lib.bo_validation import plot_validation
from lib.env import runs_dir, cases_dir, spikes_dir
from os.path import join as pj
from matplotlib import pyplot as plt

data_file = pj(cases_dir, "lrule", "stdp_mc.ssv")
data = np.loadtxt(data_file)
X = data[:,:-1]
#Y = np.asarray([data[:,-1]]).T
Y = np.asarray([np.log(np.abs(data[:,-1]))]).T

Xtr = X[:800,:]
Ytr = Y[:800]

Xtest = X[800:,:]
Ytest = Y[800:]
kernels = [
    #GPy.kern.RBF(X.shape[1], ARD=True), 
    GPy.kern.OU(X.shape[1], ARD=True), 
    #GPy.kern.Matern32(X.shape[1], ARD=True),
    #GPy.kern.Matern52(X.shape[1], ARD=True),
    #GPy.kern.ExpQuad(X.shape[1], ARD=True),
    #GPy.kern.RatQuad(X.shape[1], ARD=True),
]
mf = GPy.mappings.Linear(X.shape[1], 1)

m = None
for kern in kernels:
    m = GPy.models.GPRegression(Xtr, Ytr, kern + GPy.kern.Bias(X.shape[1]), mean_function=mf)
    m.optimize('scg', xtol=1e-6, ftol=1e-6, max_iters=200)
    means, var = m.predict(Xtest)
    se = (means-Ytest)**2
    mse = sum(se)/len(se)
    
    kern_name = repr(kern).split(" ")[0].rsplit('.',1)[1]

    f = plt.figure()        
    plt.plot(Ytest, "g-", means, "b-", se, "r-")
    plt.title("Kern: {} MSE: {}".format(kern_name, mse))
    f.savefig(pj("/home/alexeyche/dnn/runs", "{}.png".format(kern_name)))
    plt.close(f)        

    print "Kern: {} MSE: {}".format(kern_name, mse)
                
