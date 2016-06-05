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
import env
import traceback

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


class TGpyOptStrategy(TStrategy):
    def __init__(self, args):
        import GPyOpt
        import GPy
        
        self.__bounds = (0.0, 1.0)
        
        self.__asked_points = None
        self.__tellings = None

        self.__ready_to_ask = threading.Event()
        self.__got_answer = threading.Event()
        self.__got_error = threading.Event()
        self.file_cache_x = "/var/tmp/bo_cache_x.csv"
        self.file_cache_y = "/var/tmp/bo_cache_y.csv"

        self.__opt_thread = threading.Thread(target=self.__runner, args = (args,))
        self.__opt_thread.setDaemon(True)
        self.__opt_thread.start()

    def __runner(self, args):
        try:
            import GPyOpt
            import GPy
            x, y = None, None
            if os.path.exists(self.file_cache_y):
                y = np.asarray([np.loadtxt(self.file_cache_y)])
                y = y.T                
            if os.path.exists(self.file_cache_x):
                x = np.loadtxt(self.file_cache_x)
            self.__BO_parallel = GPyOpt.methods.BayesianOptimization(
                f = self.__asker,
                bounds = [(0, 1.0)]*args.dim,
                acquisition = 'EI',                 # Selects the Expected improvement
                acquisition_par = 2,                 # parameter of the acquisition function
                normalize = True,
                verbosity = True,
                model_optimize_restarts = 10,
                numdata_initial_design = 50,
                kernel = GPy.kern.OU(args.dim, ARD=True)+GPy.kern.Bias(args.dim),
                X = x,
                Y = y,
            )
            if x is None:
                np.savetxt(self.file_cache_x, self.__asked_points)
                np.savetxt(self.file_cache_y, self.__tellings)

            self.__BO_parallel.run_optimization(
                5000,                             # Number of iterations
                acqu_optimize_method = 'fast_brute',       # method to optimize the acq. function
                n_inbatch = args.jobs,                        # size of the collected batches (= number of cores)
                batch_method='predictive',                          # method to collected the batches (maximization-penalization)
                acqu_optimize_restarts = 20,                # number of local optimizers
                eps = 0.0
            )
            self.__BO_parallel.save_report(pj(env.runs_dir, "report.txt"))
        except Exception as err:
            logging.error("Got error in runner thread: {}".format(err))
            self.__got_error.set()

    
    def get_bounds(self):
        return self.__bounds
    
    def __asker(self, x):
        self.__asked_points = x
        self.__ready_to_ask.set()
        while not self.__got_answer.is_set():
            if self.__got_error.is_set():
                raise Exception("Got error in master thread")
            time.sleep(0.1)
        self.__got_answer.clear()
        return self.__tellings

    def ask(self):
        while not self.__ready_to_ask.is_set():
            if self.__got_error.is_set():
                raise Exception("Got error in master thread")
            if not self.__opt_thread.isAlive():
                return []
            time.sleep(0.1)
        return self.__asked_points

    def tell(self, x, y):
        ans = np.asarray([y])
        self.__tellings = ans.T
        self.__ready_to_ask.clear()
        self.__got_answer.set()
        
    def stop(self):
        if self.__got_error.is_set():
            raise Exception("Got error in master thread")
        return not self.__opt_thread.isAlive()
