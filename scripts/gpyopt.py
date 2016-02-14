#!/usr/bin/env python

import GPyOpt
import GPy

import logging

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

re.GlobalConfig.SimJobs = 1
re.GlobalConfig.Jobs = 1


work_dir = make_dir(pj(runs_dir, "bo_gpyopt"))

vars = read_json(re.GlobalConfig.VarSpecsFile).keys()

n_cores = 8

def evaluateSample(Xin):
    Xin = np.ndarray.tolist(Xin)
    procs = []
    answers = []
    for x in Xin:
        logging.info("Running with input {}".format(x))
        p = re.runner(x, vars, work_dir, wait=False)
        procs.append(p)
        if len(procs) >= n_cores:
            for p in procs:
                answers.append(re.communicate(p))
            procs = []
    for p in procs:
        answers.append(re.communicate(p))
    ans = -np.log(np.abs(np.asarray([answers])))
    ans = ans.T
    logging.info("Got ans: {}".format(ans))
    return ans



BO_parallel = GPyOpt.methods.BayesianOptimization(
    f=evaluateSample,
    bounds = [(0, 1.0)]*len(vars),
    acquisition = 'EI',                 # Selects the Expected improvement
    acquisition_par = 0,                 # parameter of the acquisition function
    normalize = True,
    verbosity=1,
    model_optimize_restarts=10,
    numdata_initial_design=50,
    kernel = GPy.kern.OU(len(vars), ARD=True)+GPy.kern.Bias(len(vars))
)


max_iter = 500



BO_parallel.run_optimization(
    max_iter,                             # Number of iterations
    acqu_optimize_method = 'random',       # method to optimize the acq. function
    n_inbatch = n_cores,                        # size of the collected batches (= number of cores)
    batch_method='lp',                          # method to collected the batches (maximization-penalization)
    acqu_optimize_restarts = 20,                # number of local optimizers
    eps = 1e-10
)                                # secondary stop criteria (apart from the number of iterations) 

BO_parallel.save_report(pj(work_dir, "report.txt"))
