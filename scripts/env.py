


import os
from os.path import join as pj

DNN_HOME = os.environ.get("DNN_HOME")

if DNN_HOME is None:
    raise Exception("Can't find DNN_HOME environment variable. Check that dnn is installed")


runs_dir = pj(DNN_HOME, "runs")
