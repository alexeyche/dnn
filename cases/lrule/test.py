# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:23:52 2015

@author: alexeyche
"""

from bayesopt import ContinuousGaussModel 
import random
import numpy as np

class ConcreteContinuousModel(ContinuousGaussModel):
	def __init__(self, ndim, params):
		ContinuousGaussModel.__init__(self, ndim, params)
 
	def evaluateSample(self, Xin):
         pass # mock for now

params= {}
k = "kSum(kRQISO, kSum(kMaternISO1,kMaternARD1))"
params["l_all"] = True
params['kernel_name'] = k

params["l_type"] = params.get("l_type", "empirical") # empirical fixed mcmc
params["sc_type"] = params.get("sc_type", "ml") #  # map mtl ml loocv
params['verbose_level'] = params.get("verbose_level", 1)
params['surr_name'] = params.get("surr_name", "sGaussianProcessML")
params['kernel_hp_mean'] = params.get("kernel_hp_mean", [1])
params['kernel_hp_std'] = params.get("kernel_hp_std", [1])
params
{'verbose_level': 1, 'kernel_hp_std': [1], 'surr_name': 'sGaussianProcessML', 'kernel_name': 'kSum(kMaternISO3,kSum(kRQISO,kProd(kPoly4,kConst))', 'l_type': 'empirical', 'kernel_hp_mean': [1], 'l_all': True, 'sc_type': 'ml'}

data_file = "stdp_stat.ssv"
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = data[:,-1]
nfold=5
def generate_validation_ids(n, nfold):
    test_idx = random.sample(range(n), n/nfold)
    
    train_idx = []
    j = 0
    for i in sorted(test_idx):
        train_idx += range(j, i)
        j = i+1
    train_idx += range(j, )    
    return train_idx, test_idx

test_idx, train_idx = generate_validation_ids(len(X), nfold) # test instead of train on purpose
Xtrain = X[train_idx]
Ytrain = Y[train_idx]

Xtest= X[test_idx]
Ytest= Y[test_idx]    
model = ConcreteContinuousModel(Xtrain.shape[1], params)
model.initWithPoints(Xtrain, Ytrain)