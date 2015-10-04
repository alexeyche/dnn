#!/usr/bin/env python

from scipy.optimize import differential_evolution

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

VAR_SPECS_FILE = pj(os.path.realpath(os.path.dirname(__file__)), "var_specs.json")
RUN_SIM_PY = pj(os.path.realpath(os.path.dirname(__file__)), "run_sim.py")

def scale_to(x, min, max, a, b):
    return ((b-a)*(x - min)/(max-min)) + a


def set_value_in_path(d, path, v):
    p = path[0]
    if type(d[p]) is dict:
        return set_value_in_path(d[p], path[1:], v)
    else:
        d[p] = v


def proc_vars(const, var_specs, vars, min=0.0, max=1.0):
    for k, v in vars.iteritems():
        if k not in var_specs:
            raise Exception("Can't find specs for variable {}".format(k))
        path, (a, b) = var_specs[k]
        new_v = scale_to(v, min, max, a, b)
        set_value_in_path(const, path, new_v)
    return json.dumps(const, indent=2)


def runner(x, vars, working_dir, epochs):
    working_dir = pj(working_dir, str(uuid.uuid1()))

    make_dir(working_dir)
    const_json = pj(working_dir, os.path.basename(DnnSim.CONST_JSON))
    with open(const_json, "w") as fptr:
        fptr.write(
            proc_vars(
                const = json.load(open(DnnSim.CONST_JSON))
              , var_specs = json.load(open(VAR_SPECS_FILE))
              , vars = dict(zip(vars, x))
            )
        )

    cmd = [
        RUN_SIM_PY
      , "--working-dir", working_dir
      , "--epochs", str(epochs)
      , "--const", const_json
      , "--evaluation"
      , "--slave"
    ]
    logging.info(" ".join(cmd))
    time.sleep(1000)
    p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        logging.error("process failed:")
        if stdout:
            logging.error("\n\t"+stdout)
        if stderr:
            logging.error("\n\t"+stderr)
        sys.exit(-1)

    return float(stdout.strip())

class Algo(object):
    pass

class DiffEv(Algo):
    def __init__(self, attrs):
        self.epochs = attrs.get("epochs", 10)
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
          , args = (vars, wd, self.epochs)
          , strategy = self.strategy
          , popsize = self.popsize
          , tol = self.tol
          , callback = self.callback
        )




ALGS = dict([(c.__name__, c) for c in Algo.__subclasses__()])

def main(argv):
    epi = ""
    epi += "List of variables to evolve:\n"
    for k, v in json.load(open(VAR_SPECS_FILE)).iteritems():
        epi += "\t\t{}\n\t\t\tpath: {}, range: {}-{}\n".format(k, "/".join(v[0]), v[1][0], v[1][1])
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
        '-a', '--attr', 
        required=False,
        help='Attributes for algo: "attr_name=val;attr_name2=val2"', default=""
    )
    parser.add_argument(
        '-t', '--tag', 
        required=False,
        help='Tag for run, by defailt algo choosing by himeself', default=None
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

    a = algo_cls(parse_attrs(args.attr))
    a([ v.strip() for v in args.vars.split(";") if v.strip() ], tag=args.tag)

if __name__ == '__main__':
    main(sys.argv[1:])
