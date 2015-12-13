#!/usr/bin/env python


from evolve_state import State
import sys

state = State.read_from_dir(sys.argv[1])

for v in state.vals:
    for vv in v[0]:
        for vvv in vv:
            print "{}, ".format(vvv),
    print v[1][0]

