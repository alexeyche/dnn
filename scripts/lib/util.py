
import os
import logging
import errno
import shutil
from contextlib import contextmanager
import subprocess as sub
import sys
import json
from collections import OrderedDict

def make_dir(path, delete_if_exists=False):
    if delete_if_exists and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                logging.debug("Dir {} already created. Must be the race".format(path))
    return path



def parse_attrs(attr):
    ans = {}
    attr_pairs = attr.split(";")
    for a in attr_pairs:
        a = a.strip()
        if a:
            a_v = a.split("=")
            a_v = [ v.strip() for v in a_v if v.strip() ]
            if len(a_v) != 2:
                raise Exception("Can't parse attribute string: {}".format(a))
            ans[ a_v[0] ] = a_v[1]
    return ans


def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[0].levelno
        if(levelno>=50):
            color = '\x1b[31m' # red
        elif(levelno>=40):
            color = '\x1b[31m' # red
        elif(levelno>=30):
            color = '\x1b[33m' # yellow
        elif(levelno>=20):
            color = '\x1b[32m' # green
        elif(levelno>=10):
            color = '\x1b[35m' # pink
        else:
            color = '\x1b[0m' # normal
        args[0].msg = color + args[0].msg +  '\x1b[0m'  # normal
        return fn(*args)
    return new 

@contextmanager
def pushd(newDir):
    previousDir = os.getcwd()
    os.chdir(newDir)
    yield
    os.chdir(previousDir)


def run_proc(cmd, env = {}, print_root_log_on_fail=False):
    osenv = os.environ
    osenv.update(env)

    if len(env) > 0:
        logging.info("env: {}".format(env))
    logging.info(" ".join(cmd))
    p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE, env=osenv)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        logging.error("process failed:")
        if stdout:
            logging.error("\n\t"+stdout)
        if stderr:
            logging.error("\n\t"+stderr)
        if print_root_log_on_fail:
            root_log = logging.getLogger()
            logs = [ h.stream.name for h in root_log.handlers if isinstance(h, logging.FileHandler) ]
            assert(len(logs)>0)
            print open(logs[0]).read()
        sys.exit(-1)
    return stdout


def opt_to_str(o):
    return o.lstrip("-").replace("-","_")

def str_to_opt(s):
    return "--" + s.replace("_","-")

def read_json(json_f):
    json_str = ""
    with open(json_f) as fptr:
        for l in fptr:
            l = l.split("//",1)[0]
            l = l.split("#",1)[0]
            json_str += l + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)
