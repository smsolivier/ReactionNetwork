#!/usr/bin/env python

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

def readReaction():

	skip = 60 

	f = open('starlib_v6.dat', 'r')

	nuc = ['he4', 'c12']

	for line in f:

		inc, out = readHeader(line) 

		react = inc + out 

		throw = 0 
		for i in range(len(react)):

			if not(react[i] in nuc):

				throw = 1 

		if (throw == 0):

			print(line)

		for i in range(skip):

			next(f)

	f.close()