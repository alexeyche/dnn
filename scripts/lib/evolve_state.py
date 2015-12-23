
import pickle
from os.path import join as pj

class State(object):
    FNAME = "state.p"

    def __init__(self, seed):
        self.vals = []
        self.seed = seed

    def add_val(self, X, tells):
        self.vals.append( (X, tells) )

    def dump(self, wd):
        pickle.dump(self, open(pj(wd, State.FNAME), "wb"))
   
    @staticmethod
    def read_from_dir(wd):
        return pickle.load(open(pj(wd, State.FNAME), "rb"))


