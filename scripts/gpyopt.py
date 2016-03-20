#!/usr/bin/env python

import GPyOpt
import GPy

import logging

import numpy as np
import lib.env as env
from lib.env import runs_dir, cases_dir, spikes_dir
import os
from os.path import join as pj
import subprocess as sub
import uuid

from lib.util import read_json
from lib.util import make_dir

class GlobalConfig(object):
    Epochs = 1
    Jobs = 8
    BadValue = 1.0
    SimJobs = 1
    ConfigFilename = None
    Mock = False
    NumberOfCalcutationsUpperBound = 50000


RUN_SIM_PY = pj(os.path.realpath(os.path.dirname(__file__)), "run_sim.py")




def communicate(p):
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        logging.error("process failed:")
        if stdout:
            logging.error("\n\t"+stdout)
        if stderr:
            logging.error("\n\t"+stderr)
        return GlobalConfig.BadValue
    return float(stdout.strip())

def runner(x, vars, working_dir, wait=False, id=None, min=0.0, max=1.0):
    if id is None:
        id = uuid.uuid1()
    working_dir = pj(working_dir, str(id))
    if os.path.exists(working_dir):
        raise Exception("Working dir is already exists {}!".format(working_dir))
    make_dir(working_dir)
    const_json = pj(working_dir, os.path.basename(GlobalConfig.ConstFilename))
    specs = read_json(GlobalConfig.VarSpecsFile)
    with open(const_json, "w") as fptr:
        fptr.write(
            proc_vars(
                const = read_json(GlobalConfig.ConstFilename)
              , var_specs = specs
              , vars = dict(zip(vars, x))
              , min = min
              , max = max
            )
        )
    cmd = [
        RUN_SIM_PY
      , "--working-dir", working_dir
      , "--epochs", str(GlobalConfig.Epochs)
      , "--config", config
      , "--slave"
      , "--jobs", str(GlobalConfig.SimJobs)
    ]
    logging.info(" ".join(cmd))
    if GlobalConfig.Mock:
        p = sub.Popen("sleep 1.0 && echo 1.0", shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
        if wait:
            return communicate(p)
        return p

    p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    if wait:
        return communicate(p)
    return p




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

def main(argv):
    parser = argparse.ArgumentParser(description='Gpy opt for dnn sim')
    parser.add_argument('-e', 
                        '--epochs', 
                        required=False,
                        help='Number of epochs to run', default=TDnnSim.EPOCHS,type=int)
