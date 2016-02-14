#!/usr/bin/env python

import numpy as np
import lib.env as env
from lib.env import runs_dir, cases_dir, spikes_dir
import bayesopt
from bayesopt import ContinuousGaussModel 
import os
from os.path import join as pj

from lib.bo_validation import create_model
from lib.bo_validation import get_validation_params

import run_evolve as re

from lib.util import read_json
from lib.util import make_dir


re.GlobalConfig.VarSpecsFile = pj(cases_dir, "lrule", "lrule_find_var_specs.json") 
re.GlobalConfig.ConstFilename = pj(cases_dir, "lrule", "lrule_find.json") 

re.GlobalConfig.Epochs = 5

re.GlobalConfig.AddOptions = [
    "--spike-input", 
    pj(spikes_dir, "timed_pattern_spikes.pb"),
     "--evaluation-data",
      pj(spikes_dir, "timed_pattern_spikes_test.pb")
]

re.GlobalConfig.SimJobs = 8
re.GlobalConfig.Jobs = 1


work_dir = make_dir(pj(runs_dir, "bo"))



   

vars = read_json(re.GlobalConfig.VarSpecsFile).keys()
#id = 0


class ConcreteContinuousGaussModel(ContinuousGaussModel):
    def __init__(self, ndim, params):
        assert "Gaussian" in params["surr_name"]
        ContinuousGaussModel.__init__(self, ndim, params)

    def evaluateSample(self, Xin):
        ans = re.runner(Xin, vars, work_dir, wait=True)
        return -np.log(np.abs(ans))
	

#def func(x):
#    ans = re.runner(x, vars, work_dir, wait=True)
#    return ans



data_file = pj(cases_dir, "lrule", "stdp_mc.ssv")
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = -np.log(np.abs(data[:,-1]))

#Xtr = X[:800,:]
#Ytr = Y[:800]

#Xtest = X[800:,:]
#Ytest = Y[800:]

#X = X[:100,:]
#Y = Y[:100]
D = X.shape[1]

params = get_validation_params({
    "verbose_level": 2
  , "kernel": "kMaternARD1"
  , "surr_name": "sGaussianProcessML"
#  , "crit_name": "cEI"
  , "crit_name": "cHedge(cSum(cEI,cDistance),cLCB,cPOI,cOptimisticSampling)"
  , "n_iterations" : 500
})

model = ConcreteContinuousGaussModel(D, params)

model.initWithPoints(X, Y)

#model.optimize()

#lb = np.ones((D, ))*0.0
#ub = np.ones((D, ))*1.0
# mvalue, x_out, error = bayesopt.optimize(func, X.shape[1], lb, ub, params)
#func([0.0]*D)

#model.evaluateSample([0.0]*D)
