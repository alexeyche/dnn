
import os
import sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/dnn/lib/python2.7/site-packages"))

from libdnn import generated_pb2 as gen
from libdnn import base_pb2 as base

from google.protobuf.internal import encoder

def write(p, st):
    sm = p.SerializeToString()
    l = len(sm)
    s = encoder._VarintBytes(len(sm))
    st.write(s + sm)

def writeDoubleMatrix(m, d):
    cn = base.ClassName()
    cn.class_name = "DoubleMatrix"
    cn.has_proto = True
    cn.size = 1

    dm = gen.DoubleMatrix()
    dm.nrow_v = m.shape[0]
    dm.ncol_v = m.shape[1]
    vals = []
    for j in xrange(m.shape[1]):
        for i in xrange(m.shape[0]):
            vals.append(m[i, j])
    dm.vals.extend(vals)

    with open(d, "w") as st:
        write(cn, st)
        write(dm, st)
