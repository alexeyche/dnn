#!/usr/bin/env python


from lib.evolve_state import State
import sys

state = State.read_from_dir(sys.argv[1])

for v in state.vals:
    for vv in v[0]:
        for ii, vvv in enumerate(vv):
            print "{0:.12f}".format(vvv),
    print v[1][0]

