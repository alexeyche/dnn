#!/usr/bin/env python

import logging
from os.path import join as pj
import sys
import argparse
import subprocess as sub
import os
import math
import numpy as np

from lib.util import setup_logging
from lib.evolve import TEvolveCli

setup_logging(logging.getLogger())

def run(args, runner_args):
	cli = TEvolveCli(runner_args = runner_args, bounds = (0.0, 1.0), bad_value=100000, max_running = args.jobs)    
	axis_slices = [ np.linspace(0.0, 1.0, num=int(math.pow(args.num_of_evals, 1.0/float(args.dim)))) for _ in xrange(args.dim) ]

	if len(axis_slices) == 1:
		points = axis_slices[0].reshape(1, -1).True
	else:
		points = np.vstack(np.meshgrid(*axis_slices)).reshape(args.dim, -1).T

	for p in points:
		cli.run(p)

def main(argv):
    parser = argparse.ArgumentParser(description='Tool for evolving simulations dnn')
    parser.add_argument('-j', '--jobs', required=False, type=int, help='Parallel jobs')
    parser.add_argument('-d', '--dim', required=True, type=int, help='Dimension of problem')
    parser.add_argument('-n', '--num-of-evals', required=True, type=int, help='Number of evals')

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
    run(args, runner_args)

if __name__ == '__main__':
    main(sys.argv[1:])

