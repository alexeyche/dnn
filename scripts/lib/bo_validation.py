# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:51:02 2015

@author: alexeyche
"""


from bayesopt import ContinuousGaussModel 
from bayesopt import ContinuousStudentTModel
from matplotlib import pyplot as plt
import random
from multiprocessing import Process, Queue
from collections import defaultdict
import os
from os.path import join as pj
import numpy as np

def make_dir(*a):
    if not os.path.exists(pj(*a)):
        os.mkdir(pj(*a))
    return pj(*a)

class ConcreteContinuousGaussModel(ContinuousGaussModel):
    def __init__(self, ndim, params):
        assert "Gaussian" in params["surr_name"]
        ContinuousGaussModel.__init__(self, ndim, params)

	def evaluateSample(self, Xin):
         pass # mock for now

class ConcreteContinuousStudentTModel(ContinuousStudentTModel):
    def __init__(self, ndim, params):
        assert "StudentT" in params["surr_name"]
        ContinuousStudentTModel.__init__(self, ndim, params)

	def evaluateSample(self, Xin):
         pass # mock for now


def create_model(ndim, params):
    if "Gaussian" in params["surr_name"]:
        return ConcreteContinuousGaussModel(ndim, params)
    elif "StudentT" in params["surr_name"]:
        return ConcreteContinuousStudentTModel(ndim, params)
    else:
        raise Exception("Unknown model: {}".format(params["surr_name"]))


def generate_validation_ids(n, nfold):
    test_idx = random.sample(range(n), n/nfold)
    
    train_idx = []
    j = 0
    for i in sorted(test_idx):
        train_idx += range(j, i)
        j = i+1
    train_idx += range(j, n)
    return train_idx, test_idx


def parallel(f, q):
    def wrap(*args, **kwargs):
        ret = f(*args, **kwargs)
        q.put(ret)
    return wrap
    
def run_validation(X, Y, nfold, params):
    test_idx, train_idx = generate_validation_ids(len(X), nfold) # test instead of train on purpose
    
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    
    Xtest= X[test_idx]
    Ytest= Y[test_idx]    
    model = create_model(Xtrain.shape[1], params)
    model.initWithPoints(Xtrain, Ytrain)
    
    preds = [ model.getPrediction(x) for x in Xtest ]
    means = [ p.getMean() for p in preds ] 
    se = (means-Ytest)**2
    mse = sum(se)/len(se)
    
    return Ytest, means, se, mse


def plot_validation(Ytest, means, se, mse):
    plt.plot(Ytest, "g-", means, "b-", se, "r-")
    plt.title("MSE: {}".format(mse))
                

def get_validation_params(params = {}):
    params["l_all"] = True
    params['kernel_name'] = params.get("kernel_name", "kMaternARD5")
    params["l_type"] = params.get("l_type", "empirical") # empirical fixed mcmc
    params["sc_type"] = params.get("sc_type", "ml") #  # map mtl ml loocv
    params['verbose_level'] = params.get("verbose_level", 1)
    params['surr_name'] = params.get("surr_name", "sGaussianProcessML")
    return params

def run_search(X, Y, kernels, nfold, params = {}, generate_plots=True, number_of_runs=100):
    kres = defaultdict(list)

    if generate_plots:
        plt.ioff()
    
    for k in kernels:
        params = get_validation_params(params)
        params["kernel_name"] = k
        
        procs = []
        for fi in xrange(number_of_runs):
            q = Queue()
            p = Process(target=parallel(run_validation, q), args=(X, Y, nfold, params))
            p.start()
            
            procs.append((p, q))
        
        for p, q in procs:
            p.join()
            kres[k].append(q.get())
        
        if generate_plots:
            make_dir(k)            
            fi = 0
            for res in kres[k]:
                f = plt.figure()        
                plot_validation(*res)
                f.savefig(pj(k, "{}.png".format(fi)))
                plt.close(f)        
                fi += 1
        
    kres_ag = [ (k, sum([ vr[3] for vr in v ])/len(v)) for k, v in kres.items() ]
    
    if generate_plots:
        plt.ion()

    return sorted(kres_ag, key=lambda x: x[1])
    
def combine_kernels(kernels, composite_kernels):
    k_ids = np.asarray(range(len(kernels)))
    ck_ids = np.asarray(range(len(composite_kernels)))
    axis_slices = [k_ids]*3 + [ck_ids]*2
    points = np.vstack(np.meshgrid(*axis_slices)).reshape(len(axis_slices), -1).T
    
    kernels_to_search = []
    for p in points:
        s = "{comp1}({k1}, {comp2}({k2}, {k3}))".format(
            comp1 = composite_kernels[p[3]]
          , comp2 = composite_kernels[p[4]]
          , k1 = kernels[p[0]]
          , k2 = kernels[p[1]]
          , k3 = kernels[p[2]]
        )
        kernels_to_search.append(s) 
    
    kernels_to_search += kernels
    return kernels_to_search
                
