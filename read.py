#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt 

import sys 

def chapter(label):

	inc = 0 
	out = 0 

	if (label == 1):

		inc = 1
		out = 1 

	elif (label == 2):

		inc = 1 
		out = 2 

	elif (label == 3):

		inc = 1 
		out = 3 

	elif (label == 4):

		inc = 2 
		out = 1

	elif (label == 5):

		inc = 2 
		out = 2 

	elif (label == 6):

		inc = 3 
		out = 3 

	elif (label == 7):

		inc = 2 
		out = 4 

	elif (label == 8): 

		inc = 3
		out = 1 

	elif (label == 9):

		inc = 3 
		out = 2 

	elif (label == 10):

		inc = 4 
		out = 2 

	elif (label == 11):

		inc = 1 
		out = 4 

	else:

		print('chapter not defined')
		sys.exit()

	return inc, out 

def readHeader(line):

	N = len(line) # length of the header line 

	loc = 0 

	skip = 5 

	rlabel = int(line[0:skip].strip(' '))

	nInc, nOut = chapter(rlabel)

	loc += skip 

	inc = [] 
	for i in range(nInc):

		inc.append(line[loc:loc+skip].strip(' '))

		loc += skip 

	out = [] 
	for i in range(nOut):

		out.append(line[loc:loc+skip].strip(' '))

		loc += skip 

	return inc, out

skip = 60 

f = open('starlib_v6.dat', 'r')

srows = 0 

for line in f:

	inc, out = readHeader(line) 

	if ('c12' in inc or 'c12' in out):

		print(line.strip())

	for i in range(skip):

		f.next()

f.close()