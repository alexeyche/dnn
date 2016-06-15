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
import random
import glob
import re
import time
import random

import lib.env as env
from lib.env import runs_dir

from lib.util import add_coloring_to_emit_ansi
from lib.util import pushd
from lib.util import run_proc
from lib.util import str_to_opt

THIS_FILE = os.path.realpath(__file__)
SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))

def scale_to(x, min, max, a, b):
    return ((b-a)*(x - min)/(max-min)) + a


class TDnnSim(object):
    JOBS = multiprocessing.cpu_count()
    EPOCHS = 1
    HOME = os.getenv("DNN_HOME", os.path.expanduser("~/dnn"))
    RUNS_DIR = pj(env.runs_dir, "sim")
    INSP_SCRIPT = pj(HOME, "r_scripts", "insp.R")
    USER_JSON_FILE = pj(SCRIPTS_DIR, "user.json")
    USER_JSON = json.load(open(USER_JSON_FILE))
    USER = os.environ["USER"]
    
    EVO_RE = re.compile("[\s]*(?P<Name>[a-zA-Z]+):[\s]*(?P<Given>[-.e0-9]+)[\s]*#[\s]*\[(?P<From>[-.e0-9]+),[\s]*(?P<To>[-.e0-9]+)\].*")    

    LOG_FILE_BASE = "run_sim.log"

    def __init__(self, **kwargs):
        self.current_epoch = 1

        self.old_dir = kwargs.get("old_dir", False)
        self.model = kwargs.get("model")
        if self.model is None:
            raise Exception("Model is required input to run_sim.py")

        self.config = kwargs.get("config")
        if self.config is None:
            model_dir = os.path.dirname(self.model)
            configs = glob.glob("{}/*.pb.txt".format(model_dir))
            if len(configs) == 0:
                raise Exception("Can't find any configs in model directory {}".format(model_dir))
            if len(configs) > 1:
                raise Exception("Found too much configs in model directory {}, can't choose".format(model_dir))
            self.config = configs[0]

        self.runs_dir = kwargs.get("runs_dir", self.RUNS_DIR)
        self.inspection = kwargs.get("inspection", True)
        self.working_dir = kwargs.get("working_dir", None)
        self.evaluation = kwargs.get("evaluation", True)
        self.slave = kwargs.get("slave", False)
        self.prepare_data = kwargs.get("prepare_data", False)
        self.evaluation_data = kwargs.get("evaluation_data", None)
        self.no_learning = kwargs.get("no_learning", False)
        self.evo = kwargs.get("evo", False)
        if self.evo:
            self.slave = True
            
        if self.evaluation_data:
            self.evaluation = True
        self.seed = kwargs.get("seed")
        self.connection_seed = kwargs.get("connection_seed")
        if self.seed and self.connection_seed is None:
            self.connection_seed = self.seed

        self.evaluation_script = kwargs.get("evaluation_script")
        if self.evaluation_script:
            self.inspection = False

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
        else:
            self.working_dir = os.path.realpath(os.path.expanduser(self.working_dir))
        ask = False
        force = kwargs.get("force")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        else:
            ask = not force
        
        if not self.slave:
            last = os.path.join(self.runs_dir, "..", "last")
            if os.path.exists(last) or os.path.islink(last):
                os.remove(last)
            os.symlink(self.working_dir, last)
        self.log_file = pj(self.working_dir, TDnnSim.LOG_FILE_BASE)
        fileHandler = logging.FileHandler(self.log_file, mode='w')
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        if ask:
            self.continue_in_wd()
        
        if force:
            self.clean()

        self.stat = kwargs.get("stat", False)
        self.epochs = kwargs.get("epochs", 1)
        self.jobs = kwargs.get("jobs", multiprocessing.cpu_count())
        self.insp_script = kwargs.get("insp_script", self.INSP_SCRIPT)
        self.T_max = kwargs.get("T_max", None)
        self.input_ts = kwargs.get("input_ts", None)
        self.input_spikes = kwargs.get("input_spikes", None)

        if self.input_ts is None and self.input_spikes is None:
            raise Exception("Need input time series or input spikes in model")
        if self.input_ts and self.input_spikes:
            raise Exception("Choose one of the inputs: time series or spikes, can't work with both")

        if self.input_ts:
            shutil.copy(self.input_ts, pj(self.working_dir, "input_time_series.pb")) 
        if self.input_spikes:
            shutil.copy(self.input_spikes, pj(self.working_dir, "input_spikes_list.pb")) 
                    
        wd_config = pj(self.working_dir, os.path.basename(self.config))
        if wd_config != self.config or not os.path.exists(wd_config):
            shutil.copy(self.config, self.working_dir)
            self.config = wd_config

        if self.evo:
            pars = sys.stdin.readline()
            pars = [ p.strip() for p in pars.split(" ") if p.strip() ]
            config = open(self.config).readlines()
            
            emin, emax = float(os.environ.get("EVOLVE_MIN", "0.0")), float(os.environ.get("EVOLVE_MAX", "1.0"))

            patch_left = len(pars)
            with open(self.config, "w") as fptr:
                fptr.write("# PATCHED by run_sim.py in evo mode with values (min: {}, max: {}): \n# {}\n".format(emin, emax, " ".join(pars)))
                for l in config:
                    m = TDnnSim.EVO_RE.match(l)
                    if m:
                        field = m.group("Name")
                        val = m.group("Given")
                        fr = m.group("From")
                        to = m.group("To")
                        
                        new_val = float(pars.pop(0))
                        logging.info("Found patch for field {} with value {} {} {}".format(field, new_val, fr, to))
                        if new_val > emax or new_val < emin:
                            raise Exception("Need input value between {} or {}, got {}".format(emin, emax, new_val))
                        
                        new_val = scale_to(new_val, emin, emax, float(fr), float(to))
                        new_line = ""
                        spm = re.match("^([\s]*)", l)
                        if spm:
                            new_line += spm.group(1)
                        new_line += "{}: {}      # PATCHED {}\n".format(field, new_val, patch_left-1)
                        fptr.write(new_line)
                        patch_left -= 1
                    else:
                        fptr.write(l)

            if patch_left>0:
                raise Exception("Too many variables in input to patch, {} variables left to patch".format(patch_left))

    def get_fname(self, f, ep=None):
        return os.path.join(self.working_dir, "{}_{}".format(ep if not ep is None else self.current_epoch, f)) 

    def get_opt(self, n):
        return str_to_opt(n), str(self.__dict__[n])


    def construct_default_run_cmd(self):
        cmd = [
            self.model,
            "--verbose"
        ]
        cmd += list(self.get_opt("config"))
        cmd += list(self.get_opt("jobs")) 

        if not self.T_max is None:
            cmd += ["--tmax", self.T_max]
        if self.seed:
            cmd += ["--seed", self.seed]
        if self.connection_seed:
            cmd += ["--connection-seed", self.connection_seed]
        return cmd

    def construct_eval_run_cmd(self):
        cmd = self.construct_default_run_cmd()
        cmd += [
            "--output", self.get_fname("eval_spikes.pb")
        ]
        if self.stat:
            cmd += [
                "--stat", self.get_fname("eval_stat.pb")
            ]
        model = self.get_fname("model.pb")
        if not os.path.exists(model):
            raise Exception("Can't find current model to run evaluation run")

        cmd += [
            "--load", model,
            "--no-learning"
        ]
        if self.evaluation_data:
            if self.input_ts:
                cmd += [
                    "--input-time-series", self.evaluation_data
                ]
            if self.input_spikes:
                cmd += [
                    "--input-spikes", self.evaluation_data
                ]
        return { "cmd" : cmd, "print_root_log_on_fail" : self.slave }

    def construct_cmd(self):
        cmd = self.construct_default_run_cmd()
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
        if self.no_learning:
            cmd += [
                "--no-learning"
            ]

        if self.input_spikes:
            cmd += [
                "--input-spikes", self.input_spikes
            ]
        if self.input_ts:
            cmd += [
                "--input-time-series", self.input_ts
            ]

        return { "cmd" : cmd, "print_root_log_on_fail" : self.slave, "stdout": pj(self.working_dir, "{}.log".format(self.current_epoch)) }

    def construct_eval_script_cmd(self):
        cmd = [
            self.evaluation_script,
        ]
        return {"cmd": cmd}
    
    def construct_inspect_cmd(self):
        env = {
            "T0" : os.environ.get("T0", "0"),
            "T1" : os.environ.get("T1", "2000"),
            "COPY_PICS" : "yes",
            "EP" : str(self.current_epoch),
            "OPEN_PIC" : "no",
            "SP_PIX0" : "{}".format(1024*2),
            "EVAL" : "yes" if self.evaluation else "no",
            "EVAL_JOBS" : str(self.jobs),
            "INSP_SPIKES" : "yes",
            "INSP_MODEL" : "yes",
        }
        cmd = [
              self.insp_script
        ]
        if TDnnSim.USER_JSON.get(TDnnSim.USER):
            env.update(TDnnSim.USER_JSON[TDnnSim.USER])
        env.update({
            "CONFIG" : self.config,
        })
        if env["EVAL"] in frozenset(["no", "0", "false", "False"]):
            self.evaluation = False
        return { "cmd" : cmd, "env" : env, "print_root_log_on_fail" : self.slave }


    def run(self):
        evals = []
        for self.current_epoch in xrange(self.current_epoch, self.current_epoch+self.epochs):
            logging.info("Running epoch {}:".format(self.current_epoch))
            run_proc(**self.construct_cmd())
            if self.evaluation_data:
                logging.info("running on evaluation data ...")
                run_proc(**self.construct_eval_run_cmd())
            if self.inspection:
                logging.info("inspecting ... ")
                with pushd(self.working_dir):
                    o = run_proc(**self.construct_inspect_cmd())
                    if self.evaluation:
                        evals.append(float([ line.strip("\n") for line in o.split("\n") if line.strip() ][-1]))
                        logging.info("Evaluation score: {}".format(evals[-1]))
            elif self.evaluation_script:
                with pushd(self.working_dir):
                    o = run_proc(**self.construct_eval_script_cmd())
                evals.append(float([ line.strip("\n") for line in o.split("\n") if line.strip() ][-1]))
                logging.info("Evaluation score: {}".format(evals[-1]))

        if len(evals)>0:
            final_score = sum(evals)/len(evals)
            logging.info("Final evaluation score: {}".format(final_score))
            if self.slave:
                print final_score
        logging.info("Done")

    def clean(self):
        logging.info("Cleaning {} ... ".format(self.working_dir))
        for f in os.listdir(self.working_dir):
            if f != TDnnSim.LOG_FILE_BASE:
                os.remove(os.path.join(self.working_dir, f))

    def continue_in_wd(self):
        max_ep = 0
        for f in os.listdir(self.working_dir):
            f_spl = f.split("_")
            if len(f_spl) > 1 and "model" in f:
                max_ep = max(max_ep, int(f_spl[0]))
        if max_ep>0:
            if self.slave:
                self.clean()
                return
            while True:
                ans = raw_input("%s already exists and %s epochs was done here. Continue simulation? (y/n): " % (os.path.basename(self.working_dir), max_ep))
                if ans in ["Y","y"]:
                    self.current_epoch = max_ep + 1
                    break
                elif ans in ["N", "n"]:
                    self.clean()
                    break                        
                else:
                    logging.warning("incomprehensible answer")

    def get_wd(self):
        config_hex = md5.new(open(self.config).read()).hexdigest()
        i = 0

        found = False
        while i<9999:
            self.working_dir = os.path.join(self.runs_dir, config_hex + "_%04d" % i)
            time.sleep(0.001 * random.random())
            if not os.path.exists(self.working_dir) or self.old_dir:
                found = True
                break
            i+=1
        if not found:
            raise Exception("Failed to find directory. Too much runs here ({})".format(i))
        return self.working_dir
    
def main(argv):
    parser = argparse.ArgumentParser(description='Tool for simulating dnn')
    parser.add_argument('-e', 
                        '--epochs', 
                        required=False,
                        help='Number of epochs to run', default=TDnnSim.EPOCHS,type=int)
    parser.add_argument('-j', 
                        '--jobs', 
                        required=False,
                        help='Number of parallell jobs (default: %(default)s)', default=TDnnSim.JOBS, type=int)
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
    parser.add_argument('--connection-seed',
                        required=False,
                        help='Set seed for random engine of connection builder')
    parser.add_argument('--seed',
                        required=False,
                        help='Set seed for random engine')
    parser.add_argument('-ni', 
                        '--no-insp',
                        action='store_true',
                        help='No inspection after every epoch')
    parser.add_argument('--slave',
                        action='store_true',
                        help='Run script as slave and print only evaluation score')
    parser.add_argument('--evo',
                        action='store_true',
                        help='Take from stdin parameters given by some BO optimizer')
    parser.add_argument('-f', '--force',
                        action='store_true',
                        help="Don't ask questions, just use directory")
    parser.add_argument('-r', 
                        '--runs-dir', 
                        required=False,
                        help='Runs dir (default: %(default)s)', default=TDnnSim.RUNS_DIR)
    parser.add_argument('-w', 
                        '--working-dir',
                        required=False,
                        help='Working dir (default: %%runs_dir%%/%%md5_of_config%%_%%number_of_experiment%%)')
    parser.add_argument('-c', 
                        '--config', 
                        required=False,
                        help='Path to config.pb.txt (default: run_sim.py will try to find pb.txt file in model directory)')
    parser.add_argument('-is', 
                        '--input-spikes', 
                        required=False,
                        help='Input spikes that required for model')
    parser.add_argument('-it', 
                        '--input-ts', 
                        required=False,
                        help='Input time series that required for model')
    parser.add_argument('-nev', 
                        '--no-evaluation',
                        action='store_true',
                        help='Turning on evaluation mode, where program writing score on each epoch')
    parser.add_argument('-nl', 
                        '--no-learning',
                        action='store_true',
                        help='Pass no learning flag to each sim run')
    parser.add_argument('-evd', 
                        '--evaluation-data',
                        required=False,
                        help='Run evaluation on special testing data. If not pointed evaluation will be runned on train data')
    parser.add_argument('-evs', 
                        '--evaluation-script',
                        required=False,
                        help='Run evaluation script in current working directory, inspection will be turned off')
    parser.add_argument('model', nargs=1, help="Path to model binary")
    
    args = parser.parse_args(argv)
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = {
        "config": args.config,
        "input_ts" : args.input_ts,
        "input_spikes" : args.input_spikes,
        "model" : args.model[0],
        "working_dir" : args.working_dir,
        "T_max" : args.T_max,
        "stat" : args.stat,
        "runs_dir" : args.runs_dir,
        "config" : args.config,
        "epochs" : args.epochs,
        "jobs" : args.jobs,
        "old_dir" : args.old_dir,
        "inspection" : not args.no_insp,
        "evaluation" : not args.no_evaluation,
        "slave" : args.slave,
        "evaluation_data" : args.evaluation_data,
        "force" : args.force,
        "no_learning" : args.no_learning,
        "seed" : args.seed,
        "connection_seed" : args.connection_seed,
        "evaluation_script" : args.evaluation_script,
        "evo" : args.evo,
    }
    TDnnSim(**args).run()
    
if __name__ == '__main__':
    main(sys.argv[1:])
