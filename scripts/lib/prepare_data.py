#!/usr/bin/env python

from os.path import join as pj
import os
import logging

from util import run_proc
import env
from util import str_to_opt
from util import opt_to_str


DATA_OPT = "--dst-file"
SIM_OPT = "--spike-input"

def prepare_data(prepare_data_conf):
    cmd = [pj(env.r_scripts_dir, "intercept_data_to_spikes.R")]
    
    args = []
    for k, v in prepare_data_conf.iteritems():
        opt = str_to_opt(k)
        if opt == DATA_OPT:
            v = os.path.realpath(v)
            args = [SIM_OPT, v]
            if os.path.exists(v):
                logging.info("Data file {} already exists, using it instead of new generation")
                return args
        cmd += [opt, str(v)]
    logging.info("Preparing data")
    run_proc(cmd)
    return args

