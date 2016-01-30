#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:04:19 2015

@author: alexeyche
"""

import numpy as np    

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
Y = np.log(np.abs(data[:,-1]))


nfold = 2
kernels_to_search = ["kRQISO", "kSEARD", "kMaternISO1", "kMaternISO3", "kMaternISO5", "kMaternARD1", "kMaternARD3", "kMaternARD5"]

#kernels = [
#    "kMaternARD1", "kMaternISO1",
#]
#comp_kernels = ["kSum", "kProd"]
#

#params = get_validation_params({"verbose_level": 2, "kernel": "kMaternARD1", "surr_name": "sGaussianProcessML"})

#model = create_model(X.shape[1], params)
#model.initWithPoints(X, Y)

data = np.loadtxt("/var/tmp/layer_01_start_weight.ssv")
data = np.loadtxt("/var/tmp/psp_decay.ssv"); i=6
data = np.loadtxt("/var/tmp/grid_search_stdp_ltp_ratio.ssv"); i=2
data = np.loadtxt("/var/tmp/grid_search_stdp_tau_plus.ssv"); i=0
data = np.loadtxt("/var/tmp/sigmoid_slope.ssv"); i=5
data = np.loadtxt("/var/tmp/grid_search_stdp_tau_minus.ssv"); i=1
data = np.loadtxt("/var/tmp/layer_11_start_weight.ssv"); i=4

Xtest = data[:,:-1]
Ytest = np.log(np.abs(data[:,-1]))
#GPy.kern.OU(X.shape[1], ARD=True), 
#m = GPy.models.GPRegression(Xtr, Ytr, kern + GPy.kern.Bias(X.shape[1]), mean_function=mf)
#m.optimize('scg', xtol=1e-6, ftol=1e-6, max_iters=200)
#means, var = m.predict(Xtest)

preds = [ model.getPrediction(x) for x in Xtest ]
means = [ p.getMean() for p in preds ] 
cols = cm.gray(np.linspace(0., 0.25, 2))
plt.plot(Xtest[:,i], means, "--", color=cols[1])
plt.plot(Xtest[:,i], Ytest, "-", color=cols[0])    

#for m in ["sGaussianProcessML", "sStudentTProcessJef", "sStudentTProcessNIG", "sGaussianProcess"]:
#    print "Searching for {}".format(m)    
#    res = run_search(X, Y, kernels_to_search, nfold, params = {"surr_name": m}, generate_plots=False)
#    results[m] = res

#for m, kres in results.iteritems():
##    print m
#    for k, score in kres:
#        print "\t{} {}".format(k, score)
#        
#sGaussianProcessML
#	kMaternARD1 0.0094389590004
#	kRQISO 0.0102768576038
#	kMaternISO1 0.0104187947153
#	kMaternISO3 0.0112410486542
#	kMaternISO5 0.0115246041043
#	kMaternARD3 0.0115898416816
#	kMaternARD5 0.0122498132195
#	kSEARD 0.0157078433982
#sStudentTProcessNIG
#	kRQISO 0.0103846221156
#	kMaternARD1 0.0113127458933
#	kMaternISO1 0.0118107185137
#	kMaternARD3 0.0146645319888
#	kMaternISO3 0.0155408569707
#	kSEARD 0.016342335004
#	kMaternISO5 0.0177633896141
#	kMaternARD5 0.0186057323372
#sStudentTProcessJef
#	kMaternARD1 0.00906863102819
#	kMaternISO1 0.0105977257088
#	kMaternARD3 0.0110472896123
#	kRQISO 0.011140575033
#	kMaternISO5 0.011213206942
#	kMaternISO3 0.0114194789757
#	kMaternARD5 0.0118255361048
#	kSEARD 0.0155140641631
#sGaussianProcess
#	kMaternARD1 0.011617327404
#	kMaternISO1 0.0116645625134
#	kRQISO 0.0126538553102
#	kMaternISO3 0.0156012182806
#	kMaternARD3 0.0162793005162
#	kMaternISO5 0.0182136564458
#	kMaternARD5 0.0205555507684
#	kSEARD 0.0341978422505
#        
