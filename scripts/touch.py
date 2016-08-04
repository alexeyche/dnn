#!/usr/bin/env python

import os
import sys

for f in sys.argv[1:]:
	if os.path.exists(f):
	    os.utime(f, None)
	else:
	    open(f, 'a').close()