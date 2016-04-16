#!/usr/bin/env python

import os
import sys

assert len(sys.argv) == 3
if not os.path.exists(sys.argv[2]):
    os.symlink(sys.argv[1], sys.argv[2])
