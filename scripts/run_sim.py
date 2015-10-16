#!/usr/bin/env python

import md5
import sys
import os
import argparse
import multiprocessing
import logging
import shutil
from os.path import join as pj
import json

import env

from util import add_coloring_to_emit_ansi
from util import pushd
from util import run_proc

from env import runs_dir

from prepare_data import prepare_data
from util import str_to_opt
from util import opt_to_str

THIS_FILE = os.path.realpath(__file__)

class DnnSim(object):
    JOBS = multiprocessing.cpu_count()
    EPOCHS = 1
    HOME = os.getenv("DNN_HOME", os.path.expanduser("~/dnn"))
    RUNS_DIR = pj(env.runs_dir, "sim")
    CONST_JSON = pj(HOME, "const.json")
    DNN_SIM_BIN = pj(HOME, "bin", "dnn_sim")
    INSP_SCRIPT = pj(HOME, "r_scripts", "insp.R")

    LOG_FILE_BASE = "run_sim.log"

    def __init__(self, **kwargs):
        self.current_epoch = 1

        self.old_dir = self.dget(kwargs, "old_dir", False)
        self.const_as_string = self.dget(kwargs, "const_as_string", False)
        self.const = self.dget(kwargs, "const", self.CONST_JSON)
        self.runs_dir = self.dget(kwargs, "runs_dir", self.RUNS_DIR)
        self.inspection = self.dget(kwargs, "inspection", True)
        self.working_dir = self.dget(kwargs, "working_dir", None)
        self.evaluation = self.dget(kwargs, "evaluation", False)
        self.slave = self.dget(kwargs, "slave", False)
        self.prepare_data = self.dget(kwargs, "prepare_data", False)

        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)-100s")
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.DEBUG)

        if not self.slave:
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.emit = add_coloring_to_emit_ansi(consoleHandler.emit)
            consoleHandler.setFormatter(logFormatter)
            rootLogger.addHandler(consoleHandler)

        if self.working_dir is None:
            self.working_dir = self.get_wd()

        ask = False
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        else:
            ask = True
        
        if not self.slave:
            last = os.path.join(self.runs_dir, "..", "last")
            if os.path.exists(last) or os.path.islink(last):
                os.remove(last)
            os.symlink(self.working_dir, last)
        self.log_file = pj(self.working_dir, DnnSim.LOG_FILE_BASE)
        fileHandler = logging.FileHandler(self.log_file, mode='w')
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        if ask:
            self.continue_in_wd()

        self.dnn_sim_bin = self.dget(kwargs, "dnn_sim_bin", self.DNN_SIM_BIN)
        self.add_options = self.dget(kwargs, "add_options", {})
        self.stat = self.dget(kwargs, "stat", False)
        self.epochs = self.dget(kwargs, "epochs", 1)
        self.jobs = self.dget(kwargs, "jobs", multiprocessing.cpu_count())
        self.insp_script = self.dget(kwargs, "insp_script", self.INSP_SCRIPT)
        self.T_max = self.dget(kwargs, "T_max", None)
 
        for k, v in self.add_options.items():
            if os.path.exists(v):
                shutil.copy(v, os.path.join(self.working_dir, k) + ".pb")

        wd_const = pj(self.working_dir, os.path.basename(self.const))
        if wd_const != self.const:
            if not os.path.exists(wd_const):
                shutil.copy(self.const, self.working_dir)
            self.const = wd_const



    @staticmethod
    def dget(d, n, default):
        if d.get(n) is None:
            return default
        else:
            return d[n]

    def get_opt(self, n):
        return str_to_opt(n), str(self.__dict__[n])

    def get_fname(self, f, ep=None):
        return os.path.join(self.working_dir, "{}_{}".format(ep if not ep is None else self.current_epoch, f)) 

    def construct_cmd(self):
        cmd = [
              self.dnn_sim_bin
        ]
        cmd += list(self.get_opt("const"))
        cmd += list(self.get_opt("jobs")) 
        cmd += [
            "--save", self.get_fname("model.pb"),
            "--output", self.get_fname("spikes.pb"),
        ]
        if self.stat:
            cmd += [
                "--stat", self.get_fname("stat.pb")
            ]
        prev_model = self.get_fname("model.pb", self.current_epoch-1)
        if os.path.exists(prev_model):
            cmd += [
                "--load", prev_model
            ]
        if not self.T_max is None:
            cmd += list(self.get_opt("T_max"))

        for k, v in self.add_options.items():
            cmd += [str_to_opt(k), v]

        return { "cmd" : cmd, "print_root_log_on_fail" : self.slave }
    
    def construct_inspect_cmd(self):
        env = {
            "T1" : "2000",
            "COPY_PICS" : "yes",
            "EP" : str(self.current_epoch),
            "OPEN_PIC" : "no",
            "SP_PIX0" : "{}".format(1024*2),
            "EVAL" : "yes" if self.evaluation else "no"
        }
        cmd = [
              self.insp_script
        ]
        return { "cmd" : cmd, "env" : env, "print_root_log_on_fail" : self.slave }


    def run(self):
        if self.prepare_data:
            prepare_data_conf = json.load(open(self.const)).get("prepare_data")
            with pushd(self.working_dir):
                opt, value = prepare_data(prepare_data_conf)
            self.add_options[opt_to_str(opt)] = value
        for self.current_epoch in xrange(self.current_epoch, self.current_epoch+self.epochs):
            logging.info("Running epoch {}:".format(self.current_epoch))
            run_proc(**self.construct_cmd())
            if self.inspection:
                logging.info("inspecting ... ")
                with pushd(self.working_dir):
                    o = run_proc(**self.construct_inspect_cmd())
                    if self.evaluation:
                        logging.info("Evaluation score: {}".format(o.strip()))

        logging.info("Done")
        if self.slave:
            print float(o.strip())

    def continue_in_wd(self):
        max_ep = 0
        for f in os.listdir(self.working_dir):
            f_spl = f.split("_")
            if len(f_spl) > 1 and "model" in f:
                max_ep = max(max_ep, int(f_spl[0]))
        if max_ep>0:
            def clean():
                logging.info("Cleaning %s ... " % self.working_dir)
                for f in os.listdir(self.working_dir):
                    if f != DnnSim.LOG_FILE_BASE:
                        os.remove(os.path.join(self.working_dir, f))
            if self.slave:
                clean()
                return
            while True:
                ans = raw_input("%s already exists and %s epochs was done here. Continue learning? (y/n): " % (os.path.basename(self.working_dir), max_ep))
                if ans in ["Y","y"]:
                    self.current_epoch = max_ep + 1
                    break
                elif ans in ["N", "n"]:
                    clean()
                    break                        
                else:
                    logging.warning("incomprehensible answer")


    def get_wd(self):
        const_hex = md5.new(open(self.const).read()).hexdigest()
        i = 0

        while i<1000:
            self.working_dir = os.path.join(self.runs_dir, const_hex + "_%04d" % i)
            if not os.path.exists(self.working_dir):
                break
            if self.old_dir:
                break
            i+=1
    
        return self.working_dir
    

def main(argv):
    parser = argparse.ArgumentParser(description='Tool for simulating dnn')
    parser.add_argument('-e', 
                        '--epochs', 
                        required=False,
                        help='Number of epochs to run', default=DnnSim.EPOCHS,type=int)
    parser.add_argument('-j', 
                        '--jobs', 
                        required=False,
                        help='Number of parallell jobs (default: %(default)s)', default=DnnSim.JOBS, type=int)
    parser.add_argument('-T', 
                        '--T-max', 
                        required=False,
                        help='Run only specific amount of simulation time (ms)')
    parser.add_argument('-o', 
                        '--old-dir', 
                        action='store_true',
                        help='Do not create new dir for that simulation')
    parser.add_argument('-s', 
                        '--stat',
                        action='store_true',
                        help='Save statistics')
    parser.add_argument('-ni', 
                        '--no-insp',
                        action='store_true',
                        help='No inspection after every epoch')
    parser.add_argument('-ev', 
                        '--evaluation',
                        action='store_true',
                        help='Turning on evaluation mode, where program will write only final score')
    parser.add_argument('--slave',
                        action='store_true',
                        help='Run script as slave and print only evaluation score')
    parser.add_argument('-r', 
                        '--runs-dir', 
                        required=False,
                        help='Runs dir (default: %(default)s)', default=DnnSim.RUNS_DIR)
    parser.add_argument('-w', 
                        '--working-dir',
                        required=False,
                        help='Working dir (default: %%runs_dir%%/%%md5_of_const%%_%%number_of_experiment%%)')
    parser.add_argument('-c', 
                        '--const', 
                        required=False,
                        help='Path to const.json file (default: $SCRIPT_DIR/../dnn_project/%s)' % DnnSim.CONST_JSON)
    parser.add_argument('--dnn-sim-bin', 
                        required=False,
                        help='Path to snn sim bin (default: $SCRIPT_DIR/../build/bin/%s)' % DnnSim.DNN_SIM_BIN)
    parser.add_argument('-pd', 
                        '--prepare-data',
                        action='store_true',
                        help='Run prepare data procedure')
    args, other = parser.parse_known_args(argv)

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = {
        "working_dir" : args.working_dir,
        "T_max" : args.T_max,
        "stat" : args.stat,
        "runs_dir" : args.runs_dir,
        "const" : args.const,
        "dnn_sim_bin" : args.dnn_sim_bin,
        "add_options" : {},
        "epochs" : args.epochs,
        "jobs" : args.jobs,
        "old_dir" : args.old_dir,
        "inspection" : not args.no_insp,
        "evaluation" : args.evaluation,
        "slave" : args.slave,
        "prepare_data" : args.prepare_data,
    }
    if len(other) % 2 != 0:
        raise Exception("Got not paired add options: {}".format(" ".join(other)))
    for i in range(0, len(other), 2):
        args['add_options'].update( { opt_to_str(other[i]) : other[i+1] })


    s = DnnSim(**args)
    s.run()
    
if __name__ == '__main__':
    main(sys.argv[1:])
