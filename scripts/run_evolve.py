#!/usr/bin/env python

import cma
import numpy as np

import sys
import argparse
import logging
import env
import os
import json
from os.path import join as pj
from StringIO import StringIO as sstream
import subprocess as sub
import uuid
import time
from collections import OrderedDict
import multiprocessing
import pickle
import random
import shutil

from util import make_dir
from util import parse_attrs
from util import add_coloring_to_emit_ansi

from run_sim import DnnSim


logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)-100s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.emit = add_coloring_to_emit_ansi(consoleHandler.emit)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

class GlobalConfig(object):
    Epochs = 10
    AddOptions = []
    Jobs = multiprocessing.cpu_count()/2
    BadValue = 1.0
VAR_SPECS_FILE = pj(os.path.realpath(os.path.dirname(__file__)), "var_specs.json")
RUN_SIM_PY = pj(os.path.realpath(os.path.dirname(__file__)), "run_sim.py")

def scale_to(x, min, max, a, b):
    return ((b-a)*(x - min)/(max-min)) + a


def set_value_in_path(d, path, v):
    p = path[0]
    if isinstance(d, dict):
        if d.get(p) is None:
            raise Exception("Can't find key {} in constants".format(p))
    elif isinstance(d, list):
        if len(d)<=p:
            raise Exception("Can't find key {} in constants".format(p))
    else:
        raise Exception("Got strange type in constants: {}".format(type(d)))
    if isinstance(d[p], dict) or isinstance(d[p], list):
        return set_value_in_path(d[p], path[1:], v)
    else:
        d[p] = v


def propagate_deps(const):
    weights = [ v["start_weight"]  for it in const["sim_configuration"]["conn_map"].values() for v in it ]
    mean_start_weight = sum(weights)/len(weights)
    const["globals"]["max_weight"] = 5 * mean_start_weight
    const["globals"]["mean_weight"] = mean_start_weight

def proc_vars(const, var_specs, vars, min=0.0, max=1.0):
    for k, v in vars.iteritems():
        if k not in var_specs:
            raise Exception("Can't find specs for variable {}".format(k))
        path, (a, b) = var_specs[k]
        new_v = scale_to(v, min, max, a, b)
        set_value_in_path(const, path, new_v)
    propagate_deps(const)
    return json.dumps(const, indent=2)



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

    make_dir(working_dir)
    const_json = pj(working_dir, os.path.basename(DnnSim.CONST_JSON))
    specs = json.load(open(VAR_SPECS_FILE))
    with open(const_json, "w") as fptr:
        fptr.write(
            proc_vars(
                const = json.load(open(DnnSim.CONST_JSON), object_pairs_hook=OrderedDict)
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
      , "--const", const_json
      , "--evaluation"
      , "--slave"
      , "--jobs", "1"
    ] + GlobalConfig.AddOptions
    for v in vars:
        print v
        path, range = specs[v]
        if "prepare_data" in path:
            cmd += ["--prepare-data"]
            break
    logging.info(" ".join(cmd))
    p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    if wait:
        return communicate(p)
    return p

class State(object):
    def __init__(self, seed):
        self.vals = []
        self.seed = seed

    def add_val(self, X, tells):
        self.vals.append( (X, tells) )

    def dump(self, wd):
        pickle.dump(self, open(pj(wd, "state.p"), "wb"))
   
    @staticmethod
    def read_from_dir(wd):
        return pickle.load(open(pj(wd, "state.pb"), "rb"))

class Algo(object):
    pass

class DiffEv(Algo):
    def __init__(self, attrs):
        self.strategy = attrs.get("strategy", "best1bin")
        self.popsize = attrs.get("popsize", 15)
        self.tol = attrs.get("tol", 0.01)

    def callback(self, xk, convergence):
        logging.info("Got callback with convergence {} and values:".format(convergence))
        logging.info("\t{}".format(", ".join(xk)))

    def __call__(self, vars, tag=None):
        tag = "diff_ev" if tag is None else tag
        logging.info("Running Differential evolution algo")

        wd = make_dir( pj(env.runs_dir, tag) )

        bounds = [ (0.0, 1.0) for v in vars ]

        differential_evolution(
            runner
          , bounds = bounds
          , args = (vars, wd)
          , strategy = self.strategy
          , popsize = self.popsize
          , tol = self.tol
          , callback = self.callback
        )


class CmaEs(Algo):
    def __init__(self, attrs):
        self.max_bound = attrs.get("max_bound", 10)
        self.min_bound = attrs.get("min_bound", 0)
        self.popsize = attrs.get("popsize", 15)
        self.sigma = attrs.get("sigma", 2)
        self.seed = attrs.get("seed", random.randint(0, 65535))

    @staticmethod
    def wait_pool(pool, ans_list):
        while True:
            for pi, (id, p) in enumerate(pool):
                if not p.poll() is None:
                    ans_list.append( (id, communicate(p)) )
                    del pool[pi]
                    return
                time.sleep(0.5)

    def __call__(self, vars, tag=None):
        tag = "cma_es" if tag is None else tag
        wd = pj(env.runs_dir, tag)
        state = None
        if os.path.exists(wd):
            while True:
                ans = raw_input("%s already exists. Continue learning? (y/n): " % (wd))
                if ans in ["Y","y"]:
                    state = State.read_from_dir(wd) 
                    break
                elif ans in ["N", "n"]:
                    state = State(self.seed)
                    shutil.rmtree(wd)
                    make_dir(wd)
                    break                        
                else:
                    logging.warning("incomprehensible answer")

        make_dir(wd)
        state.dump(wd)

        rng = np.random.RandomState(state.seed)
        start_vals = rng.random_sample(len(vars))


        es = cma.CMAEvolutionStrategy(
            start_vals
          , self.sigma
          , { 
              'bounds' : [ 
                self.min_bound
              , self.max_bound 
             ], 
              'popsize' : self.popsize 
            , 'seed' : state.seed
          }
        )
        for X, tells in state.vals:
            X = es.ask()
            es.tell(X, tells)
        id = len(state.vals)
        while not es.stop():
            X = es.ask()
            tells = []
            pool = []
            Xw = X
            while len(Xw)>0:
                p = runner(Xw[0], vars, wd, wait=False, id=id, min=self.min_bound, max=self.max_bound)
                pool.append( (id, p) )
                if len(pool)>=GlobalConfig.Jobs:
                    self.wait_pool(pool, tells)
                id+=1
                Xw = Xw[1:]
            tells = [ out for id, out in sorted(tells, key=lambda x: x[0]) ]
            es.tell(X, tells)
            es.disp()
            state.add_val(X, tells)
            state.dump(wd)

ALGS = dict([(c.__name__, c) for c in Algo.__subclasses__()])

def main(argv):
    epi = ""
    epi += "List of variables to evolve:\n"
    for k, v in json.load(open(VAR_SPECS_FILE), object_pairs_hook=OrderedDict).iteritems():
        epi += "\t\t{}\n\t\t\tpath: {}, range: {}-{}\n".format(k, "/".join([ str(subv) for subv in v[0]]), v[1][0], v[1][1])
    epi += "List of algorithms:\n"
    for a in ALGS:
        epi += "\t{}\n".format(a)
        inst = ALGS[a]({})
        def_attrs = dict([ (a, getattr(inst, a)) for a in dir(inst) if not a.startswith("__") and not callable(getattr(inst, a)) ])
        for k, v in def_attrs.iteritems():
            epi += "\t\t{}, default: {}\n".format(k, v) 

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Tool for evolving simulations dnn',
        epilog = epi
    )
    parser.add_argument(
        '-v', '--vars', 
        required=True,
        help='Variables included in evolving, separated by ;'
    )
    parser.add_argument(
        '-e', '--epochs', 
        required=False,
        help='Epochs to run sim on each run', default=10
    )
    parser.add_argument(
        '-a', '--attr', 
        required=False,
        help='Attributes for algo: "attr_name=val;attr_name2=val2"', default=""
    )
    parser.add_argument(
        '-t', '--tag', 
        required=False,
        help='Tag for run, by defailt algo choosing by himself', default=None
    )
    parser.add_argument(
        'algo_name', nargs=1
    )
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    args, other = parser.parse_known_args(argv)
    algo_cls = ALGS.get(args.algo_name[0])    
    if algo_cls is None:
        raise Exception("Can't find algo with then name {}".format(args.algo_name[0]))

    GlobalConfig.Epochs = args.epochs
    GlobalConfig.AddOptions = other
    a = algo_cls(parse_attrs(args.attr))
    a([ v.strip() for v in args.vars.split(";") if v.strip() ], tag=args.tag)

if __name__ == '__main__':
    main(sys.argv[1:])
