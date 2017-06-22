#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class Nuclide:
	''' store name and mass fraction of a nuclide ''' 

	def __init__(self, name, y):

		self.name = name 

		self.y = y 

class Network:
	''' hold collection of nuclides for a single grid point ''' 

	def __init__(self):

		self.nuc = [] # list of nuclides in network

		self.N = 0 # number of nuclides in network 

	def updateNuc(self, name, y):
		''' update a mass fraction in the list
			add nuclide if not already there 
		''' 

		# check if nuclide present in nuc list 
		nucid = -1 # store location of nuclide in nuc

		for i in range(len(self.nuc)):

			if (self.nuc[i].name == name):

				nucid = i

		if (nucid == -1): # if new nuclide 

			self.nuc.append(Nuclide(name, y)) # create new nuclide object 

			self.N += 1 # update number of nuclides 

		else: # if already present 

			self.nuc[nucid].y = y # update mass fraction 

	def getY(self, nuc):
		''' search for nuclide nuc in list and return the mass fraction ''' 

		ind = -1 

		for i in range(self.N):

			if (self.nuc[i].name == nuc):

				ind = i 

		if (ind != -1):

			return self.nuc[ind].y

		else:

			print('--- {} not found in network --- '.format(nuc))

class Reaction:

	def __init__(self, rate, inc, out):

		self.rate = rate 
		self.inc = inc 
		self.out = out 

		self.Nin = len(inc)
		self.Nout = len(out) 

	def getRate(self, name):

		if (name in self.inc):

			return -self.rate 

		else:

			return self.rate 

net = Network() 

net.updateNuc('n', 1)
net.updateNuc('u235', 1)
net.updateNuc('u236', 0)
net.updateNuc('g', 0)

reaction = Reaction(1, ['n', 'u235'], ['u236','g'])

# build R 
R = np.zeros(net.N)
for i in range(net.N):

	R[i] = reaction.getRate(net.nuc[i].name)
	for j in range(reaction.Nin):

		R[i] *= net.getY(reaction.inc[j])

# build jacobian 
J = np.zeros((net.N, net.N))
for i in range(net.N):

	for j in range(net.N):

		J[i,j] = 

