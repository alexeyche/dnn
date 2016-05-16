#!/usr/bin/env python

import logging
import threading
import signal
from os.path import join as pj
import subprocess as sub
import os
import time
import numpy as np
import sys

class TEvolveCli(object):
    def __init__(self, runner_args, bounds, bad_value = 10.0, max_running = 8):
        self.runner_args = runner_args
        self.stop = threading.Event()
        self.bad_value = bad_value
        self.calculations = []
        self.max_running = max_running
        self.bad_value = bad_value
        signal.signal(signal.SIGINT, self.interrupt)
        self.points = []
        self.run_env = os.environ.copy()
        self.run_env["EVOLVE_MIN"] = str(bounds[0])
        self.run_env["EVOLVE_MAX"] = str(bounds[1])

    def __communicate(self, p):
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            if self.bad_value:
                return self.bad_value
            raise Exception("Found failed command: \n{}\n{}".format(stdout, stderr))
        if stderr:
            if self.bad_value:
                return self.bad_value
            raise Exception("Found failed command: \nstdout:\n{}\nstderr:\n{}".format(stdout, stderr))
        return float(stdout.strip())

    def sync_finished(self):
        finished_calcs, non_finished_calcs = [], []
        for point, proc in self.calculations:
            if proc.poll() is None:
                non_finished_calcs.append( (point, proc) )
            else:
                finished_calcs.append( (point, proc) )
        self.calculations = non_finished_calcs

        for point, proc in finished_calcs:
            self.points.append( (point, self.__communicate(proc)) )

    def sync(self):
        while len(self.calculations) > 0:
            self.sync_finished()
            time.sleep(1.0)
    
    def interrupt(self, int_signal, frame):
        logging.info("Got interrupted, killing all runs")
        for _, p in self.calculations:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        sys.exit(0)

    def stopped(self):
        return self.stop.is_set()

    def add_calculation(self, point, proc):
        while len(self.calculations)>=self.max_running:
            self.sync_finished()
            time.sleep(1.0)

        self.calculations.append( (point, proc) )

    def run(self, x):
        p = sub.Popen(self.runner_args, stdin = sub.PIPE, stdout = sub.PIPE, stderr = sub.PIPE, preexec_fn = os.setsid, env = self.run_env)
        inp = " ".join([ str(xv) for xv in x ])
        logging.info("Running with\n{}\nwith {}".format(inp, " ".join(self.runner_args)))
        p.stdin.write(inp + "\n")
        self.add_calculation(x, p)


class TStrategy(object):
    def get_bounds(self):
        raise NotImplementedError("Need override in inherited class")

    def stop(self):
        raise NotImplementedError("Need override in inherited class")
        
    def ask(self):
        raise NotImplementedError("Need override in inherited class")
        
    def tell(self, x, y):
        raise NotImplementedError("Need override in inherited class")
        

class TCMAStrategy(TStrategy):
    def __init__(self, args, start=None, dev = 2.0, popsize = 20):
        if start is None:
            start = 10.0*np.random.ranf(args.dim)
        self.__bounds = (0, 10.0)
        import cma
        self.__es =  cma.CMAEvolutionStrategy(start, dev, 
            {'bounds': [self.__bounds[0], self.__bounds[1]], 'popsize': popsize, "verb_disp": 1, "verbose": True}
        )

    def get_bounds(self):
        return self.__bounds

    def stop(self):
        return self.__es.stop()

    def ask(self):
        return self.__es.ask()

    def tell(self, x, y):
        self.__es.tell(x, y)
