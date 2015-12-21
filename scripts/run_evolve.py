#!/usr/bin/env python

import numpy as np
import sys
import argparse
import logging
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
import re

import lib.cma as cma
import lib.env as env
from lib.util import read_json
from lib.util import make_dir
from lib.util import parse_attrs
from lib.util import add_coloring_to_emit_ansi
from lib.evolve_state import State

from run_sim import DnnSim


logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)-100s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler(sys.stderr)
consoleHandler.emit = add_coloring_to_emit_ansi(consoleHandler.emit)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

class GlobalConfig(object):
    Epochs = 1
    AddOptions = []
    Jobs = 8
    BadValue = 1.0
    SimJobs = 1
    ConstFilename = DnnSim.CONST_JSON
    VarSpecsFile = pj(os.path.realpath(os.path.dirname(__file__)), "var_specs.json")
    Mock = False
    NumberOfCalcutationsUpperBound = 50000


RUN_SIM_PY = pj(os.path.realpath(os.path.dirname(__file__)), "run_sim.py")


class Distribution(object):
    @staticmethod
    def parse_distribution(s, val=None):
        if isinstance(s, basestring):
            m = re.match("Exp\((.*?)\)", s)
            if m:
                return ExpDistribution(float(m.group(1)) if val is None else val)
        return None

class ExpDistribution(object):
    def __init__(self, r):
        self.rate = r

    def __getitem__(self, idx):
        assert(idx == 0)
        return self.rate

    def __str__(self):
        return "Exp({})".format(self.rate)

    def __repr__(self):
        return str(self)


def scale_to(x, min, max, a, b):
    return ((b-a)*(x - min)/(max-min)) + a

def proc_element(d, p):
    if isinstance(d, dict):
        if d.get(p) is None:
            raise Exception("Can't find key {} in constants".format(p))
        return d
    elif isinstance(d, list):
        if len(d)<=p:
            raise Exception("Can't find key {} in constants".format(p))
        return d
    elif isinstance(d, basestring):
        return Distribution.parse_distribution(d)
    else:        
        raise Exception("Got strange type in constants: {}".format(type(d)))

def set_value_in_path(d, path, v):
    p = path[0]
    d = proc_element(d, p)
    if isinstance(d[p], dict) or isinstance(d[p], list):
        return set_value_in_path(d[p], path[1:], v)
    else:
        distr = Distribution.parse_distribution(d[p], v)
        if distr:
            d[p] = str(distr)
        else:
            d[p] = v

def get_value_in_path(d, path):
    p = path[0]
    d = proc_element(d, p)
    if isinstance(d[p], dict) or isinstance(d[p], list) or Distribution.parse_distribution(d[p]):
        return get_value_in_path(d[p], path[1:])
    else:
        return d[p]




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


def get_vars(const, var_specs, vars, min=0.0, max=1.0):
    var_values = []
    for k in vars:
        if k not in var_specs:
            raise Exception("Can't find specs for variable {}".format(k))
        path, (a, b) = var_specs[k]
        v = get_value_in_path(const, path)        
        scaled_v = scale_to(v, a, b, min, max)
        var_values.append(scaled_v)    
    return dict(zip(vars, var_values))

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
      , "--const", const_json
      , "--slave"
      , "--jobs", str(GlobalConfig.SimJobs)
    ] + GlobalConfig.AddOptions
    for v in vars:
        path, range = specs[v]
        if "prepare_data" in path:
            cmd += ["--prepare-data"]
            break
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

def lhs_sample(n, rng):
    return (np.asarray(random.sample(range(1,n+1), n)) - rng.random_sample(n))/float(n)
    

class Algo(object):
    def create_workdir(self, wd):
        state = None
        if os.path.exists(wd):
            while True:
                ans = raw_input("%s already exists. Continue learning? (y/n): " % (wd))
                if ans in ["Y","y"]:
                    state = State.read_from_dir(wd) 
                    break
                elif ans in ["N", "n"]:
                    logging.warning("Deleting {}".format(wd))
                    shutil.rmtree(wd)
                    make_dir(wd)
                    break                        
                else:
                    logging.warning("incomprehensible answer")
        else:
            make_dir(wd)
        return wd, state

    @staticmethod
    def wait_pool(pool, ans_list):
        while True:
            for pi, (id, p) in enumerate(pool):
                if not p.poll() is None:
                    ans_list.append( (id, communicate(p)) )
                    del pool[pi]
                    return
                time.sleep(0.5)

    @staticmethod
    def dump_state(wd, state, asks, tells, pool):
        asks_d = dict(asks)
        X = list()
        tells_current = list()
        finished_ids = dict()
        for finished_id, tell in [ (finished_id, tell) for finished_id, tell in sorted(tells, key=lambda x: x[0]) ]:
            X.append(asks_d[finished_id])
            del asks_d[finished_id]
            tells_current.append(tell)
            finished_ids[finished_id] = True
        state.add_val(X, tells_current)
        state.dump(wd)

        pool = [ (idp, p) for idp, p in pool if idp not in finished_ids ]
        tells = [ (idp, t) for idp, t in tells if idp not in finished_ids ]
        return state, asks_d.items(), tells, pool


class MonteCarlo(Algo):
    def __init__(self, attrs):
        self.number = int(attrs.get("number", 1000))
        self.seed = attrs.get("seed", random.randint(0, 65535))
        self.max_bound = attrs.get("max_bound", 1)
        self.min_bound = attrs.get("min_bound", 0)


    def __call__(self, vars, tag=None):
        wd, state = self.create_workdir(
            pj(
                env.runs_dir
              , "mc" if tag is None else tag
            )
        )
        if state is None:
            state = State(self.seed)
        state.dump(wd)

        rng = np.random.RandomState(state.seed)

        asks, tells, pool = [], [], []
        run_ids = range(sum([ len(v[1]) for v in state.vals ]), self.number)
        dim_size = len(vars)
        X = np.zeros((len(run_ids), dim_size))
        for dim_idx in xrange(dim_size):
            X[:,dim_idx] = self.min_bound + self.max_bound*lhs_sample(len(run_ids), rng)

        for x_id, run_id in enumerate(run_ids):
            x = X[x_id, :]
            pool.append( (run_id, runner(x, vars, wd, wait=False, id=run_id, min=self.min_bound, max=self.max_bound)) )
            asks.append( (run_id, x) )

            if len(pool)>=GlobalConfig.Jobs:
                Algo.wait_pool(pool, tells)
                state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)

        while len(pool)>0:
            Algo.wait_pool(pool, tells)
            state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)


class CmaEs(Algo):
    def __init__(self, attrs):
        self.max_bound = attrs.get("max_bound", 10)
        self.min_bound = attrs.get("min_bound", 0)
        self.popsize = attrs.get("popsize", 15)
        self.sigma = attrs.get("sigma", 2)
        self.seed = attrs.get("seed", random.randint(0, 65535))


    def __call__(self, vars, tag=None):
        wd, state = self.create_workdir(
            pj(
                env.runs_dir
              , "cma_es" if tag is None else tag
            )
        )
        if state is None:
            state = State(self.seed)
        state.dump(wd)

        #start_vals = get_vars(
        #    const = read_json(GlobalConfig.ConstFilename)
        #  , var_specs = read_json(GlobalConfig.VarSpecsFile)
        #  , vars = vars
        #  , min = self.min_bound
        #  , max = self.max_bound
        #)
        rng = np.random.RandomState(state.seed)
        start_vals = self.min_bound + self.max_bound*rng.random_sample(len(vars))

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
        id = sum([ len(v[1]) for v in state.vals ])
        while not es.stop():
            X = es.ask()
            asks, tells, pool = [], [], []

            tells, pool = [], []
            for Xi in X:
                p = runner(Xi, vars, wd, wait=False, id=id, min=self.min_bound, max=self.max_bound)
                pool.append( (id, p) )
                id+=1
                if len(pool)>=GlobalConfig.Jobs:
                    Algo.wait_pool(pool, tells)
                    state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)

            while len(pool)>0:
                self.wait_pool(pool, tells)

            state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)

            tells = [ out for _, out in sorted(tells, key=lambda x: x[0]) ]
            es.tell(X, tells)
            es.disp()

class GridSearch(Algo):
    def __init__(self, attrs):
        self.max_bound = attrs.get("max_bound", 1)
        self.min_bound = attrs.get("min_bound", 0)
        self.step = float(attrs.get("step", 0.1))
        self.freeze_point = attrs.get("freeze_point", None)
        self.non_freeze_vars = attrs.get("non_freeze_vars", None)

    def __call__(self, vars, tag=None):
        wd, state = self.create_workdir(
            pj(
                env.runs_dir
              , "grid_search" if tag is None else tag
            )
        )
        if state is None:
            state = State(0) # doesn't matter
        state.dump(wd)
        if self.freeze_point:
            self.freeze_point = [ float(p) for p in self.freeze_point.split(" ") if p.strip() ]
            if self.non_freeze_vars is None:
                raise Exception("Got freeze point but freeze variables are not defined")

        if self.non_freeze_vars:
            self.non_freeze_vars = [ v.strip() for v in self.non_freeze_vars.split(" ") if v.strip() ]
        else:
            self.non_freeze_vars = vars

        dim_of_problem = len(self.non_freeze_vars)
        number_of_dim_slice = (self.max_bound - self.min_bound)/self.step
        axis_slices = [ np.linspace(self.min_bound, self.max_bound, num=number_of_dim_slice) for _ in xrange(dim_of_problem) ]
        if len(axis_slices) == 1:
            points = axis_slices[0].reshape(1, -1).T
        else:
            points = np.vstack(np.meshgrid(*axis_slices)).reshape(dim_of_problem, -1).T
        number_of_steps = len(points)
        if number_of_steps > GlobalConfig.NumberOfCalcutationsUpperBound:
            raise Exception("There are a lot of calculations ({}). Reconsider your setup".format(number_of_steps))

        nsteps_done = sum([ len(v[1]) for v in state.vals ])
        points = points[nsteps_done:]
        asks, tells, pool = [], [], []
        for id, point in enumerate(points):
            if self.freeze_point:
                x = list(self.freeze_point)
                for vi, v in enumerate(self.non_freeze_vars):
                    try:
                        var_idx = vars.index(v)
                    except ValueError:
                        raise Exception("Can't find non freeze var {} in specification".format(v))
                    x[var_idx] = point[vi]
            else:
                x = point
            pool.append( (id, runner(x, vars, wd, wait=False, id=id, min=self.min_bound, max=self.max_bound)) )
            asks.append( (id, x) )
            if len(pool)>=GlobalConfig.Jobs:
                Algo.wait_pool(pool, tells)
                state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)
                
        while len(pool)>0:
            Algo.wait_pool(pool, tells)
            state, asks, tells, pool = Algo.dump_state(wd, state, asks, tells, pool)


class SimpleRunner(Algo):
    def __init__(self, attrs):
        self.max_bound = attrs.get("max_bound", 1)
        self.min_bound = attrs.get("min_bound", 0)
        self.id = attrs.get("id", None)

    def __call__(self, vars, tag=None):
        if tag is None:
            raise Exception("SimpleRunner need a tag")
        if self.id is None:
            raise Exception("id must be set for runner")

        wd = make_dir(pj(env.runs_dir, tag))
        
        d = np.loadtxt(sys.stdin)
        p = runner(d, vars, wd, wait=True, id=self.id, min=self.min_bound, max=self.max_bound)
        print p

ALGS = dict([(c.__name__, c) for c in Algo.__subclasses__()])

def main(argv):
    epi = ""
    epi += "List of variables to evolve:\n"
    for k, v in read_json(GlobalConfig.VarSpecsFile).iteritems():
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
        required=False,
        help='Variables included in evolving, separated by ;. Or use all variables; Or if it is a file use all variables from that file'
    )
    parser.add_argument(
        '-e', '--epochs', 
        required=False,
        help='Epochs to run sim on each run', default=1
    )
    parser.add_argument(
        '-j', '--jobs', 
        required=False, type=int,
        help='Number of parallel jobs for evolving procedure', default=multiprocessing.cpu_count()
    )
    parser.add_argument(
        '-a', '--attr', 
        required=False,
        help='Attributes for algo: "attr_name=val;attr_name2=val2"', default=""
    )
    parser.add_argument(
        '-sj', '--sim-jobs', 
        required=False,
        help='Sim jobs', default=1
    )
    parser.add_argument(
        '-t', '--tag', 
        required=False,
        help='Tag for run, by defailt algo choosing by himself', default=None
    )
    parser.add_argument(
        '-c', '--const', 
        required=False,
        help='Constants to work with, default %(default)s', default=DnnSim.CONST_JSON
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
    GlobalConfig.SimJobs = args.sim_jobs
    GlobalConfig.Jobs = args.jobs
    GlobalConfig.ConstFilename = args.const
    a = algo_cls(parse_attrs(args.attr))
    vars_str = None
    if args.vars:
        if os.path.isfile(args.vars):
            GlobalConfig.VarSpecsFile = args.vars
        else:
            vars_str = args.vars
    vars = [ v.strip() for v in vars_str.split(";") if v.strip() ] if vars_str else read_json(GlobalConfig.VarSpecsFile).keys()

    a(vars, tag=args.tag)

if __name__ == '__main__':
    main(sys.argv[1:])
