import logging
import sys
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

import env
sys.path.insert(0, env.include_dir)

import dnn.protos.config_pb2 as config_pb


_dnnlib = np.ctypeslib.load_library('libdnn', env.lib_dir)

_dnnlib.write_time_series.restype = None
_dnnlib.write_time_series.argtypes = [
    ndpointer(ndim=2, flags='CONTIGUOUS,ALIGNED'), 
    ct.c_int, 
    ct.c_int, 
    ct.c_char_p, 
    ct.c_char_p
]

_dnnlib.run_iaf_network.restype = None
_dnnlib.run_iaf_network.argtypes = [
    ct.c_char_p, # Config
    ndpointer(ndim=2, flags='CONTIGUOUS,ALIGNED'),  # Input 
    ct.c_int,  # NRows
    ct.c_int,  # NCols
    ct.c_char_p # dst file with spikes
]


def write_time_series(arr, dst_file, lab="label"):
    assert len(arr.shape) == 2
    (nrows, ncols) = arr.shape
    _dnnlib.write_time_series(arr, nrows, ncols, lab, dst_file)

def run_iaf_network(inp, dst_file, dt = 0.5, tau_mem = 10, tau_ref = 2, threshold = 0.25):
    assert len(inp.shape) == 2
    (nrows, ncols) = inp.shape
    
    cfg = config_pb.TConfig()
    cfg.SimConfiguration.Dt = dt
    layer = cfg.Layer.add()
    
    iaf = layer.IntegrateAndFire.add()
    iaf.TauMem = tau_mem
    iaf.TauRef = tau_ref

    determ = layer.Determ.add()
    determ.Threshold = threshold
    logging.info("Ready to run network with config\n{}\n data {}x{}, saving result in {}".format(cfg, nrows, ncols, dst_file))
    _dnnlib.run_iaf_network(str(cfg), inp, nrows, ncols, dst_file)
    