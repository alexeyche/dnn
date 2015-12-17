# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:04:19 2015

@author: alexeyche
"""


import numpy as np
from bayesopt import ContinuousModel

class ConcreteContinuousModel(ContinuousModel):
	def __init__(self, ndim, params):
		ContinuousModel.__init__(self, ndim, params)

	def evaluateSample(self, Xin):
         pass # mock for now

data_file = "/home/alexeyche/dnn/runs/mc/state.ssv"
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = data[:,-1]

X = X[:500]
Y = Y[:500]

params = {}
params["l_all"] = True
params["l_type"] = "empirical" # discrete fixed mcmc
params["sc_type"] = "map" #  # map mtl ml loocv
params['verbose_level'] = 2
params['kernel_name'] = "kMaternARD5"
params['kernel_hp_mean'] = [1] 
params['kernel_hp_std'] = [5]

model = ConcreteContinuousModel(X.shape[1], params)
model.initWithPoints(X, Y)
