
import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

import env

_dnnlib = np.ctypeslib.load_library('libdnn', env.lib_dir)

_dnnlib.write_time_series.restype = None
_dnnlib.write_time_series.argtypes = [ndpointer(ndim=2, flags='CONTIGUOUS,ALIGNED'), ct.c_int, ct.c_int, ct.c_char_p, ct.c_char_p]


def write_time_series(arr, dst_file, lab="label"):
	assert len(arr.shape) == 2
	(nrows, ncols) = arr.shape
	_dnnlib.write_time_series(arr, nrows, ncols, lab, dst_file)