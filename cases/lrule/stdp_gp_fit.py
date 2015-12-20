# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:04:19 2015

@author: alexeyche
"""


from validation_lib import run_search
from validation_lib import run_validation
import numpy as np    

data_file = "stdp_stat2.ssv"
data = np.loadtxt(data_file)
X = data[:,:-1]
Y = data[:,-1]

kernels_to_search = ["kRQISO", "kSEARD", "kMaternISO1", "kMaternISO3", "kMaternISO5", "kMaternARD1", "kMaternARD3", "kMaternARD5"]

#kernels = [
#    "kMaternARD1", "kMaternISO1",
#]
#comp_kernels = ["kSum", "kProd"]
#


res = run_search(X, Y, kernels_to_search, 2)
print res

#[('kMaternISO1', 0.004636857977467183), ('kMaternARD1', 0.0047115095486403085), ('kRQISO', 0.0047483236262090764), ('kMaternISO3', 0.0047903690584017358), ('kMaternARD3', 0.0051533178891543241), ('kMaternISO5', 0.0051928755612535597), ('kMaternARD5', 0.0055567855747718446), ('kSEARD', 0.0076824485862581945)]
