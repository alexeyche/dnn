#!/usr/bin/env python

import logging
from os.path import join as pj
import sys
import argparse
import subprocess as sub
import os

from lib.util import setup_logging
from lib.evolve import TCMAStrategy, TGpyOptStrategy, TEvolveCli


setup_logging(logging.getLogger())

START = "1.1832974296 2.9283499075 9.9628038855 7.12223105109 1.34217607074 0.77473632932"

def evolve(args, runner_args):
    es = TCMAStrategy(
        args, 
        popsize = 50,
        start = [ float(v) for v in START.split(" ") ], 
        dev = 0.5
    )
    
    #es = TGpyOptStrategy(args)

    while not es.stop():
        cli = TEvolveCli(runner_args = runner_args, bounds = es.get_bounds(), bad_value=100000, max_running = args.jobs)
        
        for x in es.ask():
            cli.run(x)
        cli.sync()

        es.tell([ c[0] for c in cli.points ], [ c[1] for c in cli.points ])

def main(argv):
    parser = argparse.ArgumentParser(description='Tool for evolving simulations dnn')
    parser.add_argument('-d', '--dim', required=True, type=int, help='Dimension of problem')
    parser.add_argument('-n', '--num-of-evals', required=False, type=int, help='Number of evals')
    parser.add_argument('-j', '--jobs', required=False, type=int, help='Parallel jobs')
    
    if len(argv) == 0 or argv[0] in frozenset(["--help", "-h"]):
        parser.print_help()
        sys.exit(1)

    try:
        i = argv.index("--")
        runner_args = argv[i+1:]
        argv = argv[:i]
    except ValueError:
        raise Exception("Need -- as separator of runner script")
    
    
    args = parser.parse_args(argv)
    runner_args = [ os.path.expanduser(a) for a in runner_args ]
    runner_args = [ os.path.realpath(a) if os.path.exists(a) else a for a in runner_args ]
    evolve(args, runner_args)

if __name__ == '__main__':
    main(sys.argv[1:])

