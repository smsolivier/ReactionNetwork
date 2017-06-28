#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import os 

import time 

N = 2**np.arange(10, 30)

cuda32 = np.zeros(len(N)) 
cuda128 = np.zeros(len(N))
serial = np.zeros(len(N))

for i in range(len(N)):

	start = time.time() 
	os.system('./cuda32 {}'.format(N[i]))
	end = time.time()

	cuda32[i] = end - start 

	start = time.time() 
	os.system('./cuda128 {}'.format(N[i]))
	end = time.time() 
	cuda128[i] = end - start 

	start = time.time() 
	os.system('./serial {}'.format(N[i]))
	end = time.time() 

	serial[i] = end - start 

plt.loglog(N, serial, '-o', label='serial')
plt.loglog(N, cuda32, '-o', label='CUDA32')
plt.loglog(N, cuda128, '-o', label='CUDA128')
plt.legend(loc='best')

plt.show() 