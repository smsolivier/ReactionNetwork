#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d 
from scipy.special import binom

from read import * 


class Nuclide:
	''' store name and mass fraction of a nuclide ''' 

	def __init__(self, name, y):

		self.name = name 

		self.y = y 

class Network:
	''' hold collection of nuclides for a single grid point ''' 

	def __init__(self, T, rho):

		self.nuc = [] # list of nuclides in network

		self.name = [] 

		self.y = np.array([]) 

		self.N = 0 # number of nuclides in network 

		self.reaction = [] 

		self.T = T 
		self.rho = rho

		self.rateLimit = None

	def setRateLimit(self, rateLimit):

		self.rateLimit = rateLimit

	def buildNetwork(self):

		f = open('starlib_v6.dat', 'r')

		skip = 60 

		for line in f: 

			inc, out = readHeader(line)

			react = inc + out 

			throw = 0 

			for i in range(len(react)):

				if not(react[i] in self.name):

					throw = 1 

			if (throw == 0):

				print(line.strip())

				T = [] 
				rate = [] 

				for i in range(skip):

					vals = next(f).split()

					T.append(float(vals[0]))
					rate.append(float(vals[1]))

				self.reaction.append(Reaction(np.array(T), np.array(rate), inc, out))

			else:

				for i in range(skip):

					next(f)

		print('Number of Reactions =', len(self.reaction))

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
			self.name.append(name)
			self.y = np.append(self.y, y)

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

	def getIndex(self, lst):

		out = [] 
		for i in range(len(lst)):

			out.append(self.name.index(lst[i]))

		return out 

	def buildR(self):

		R = np.zeros(self.N)

		for k in range(len(self.reaction)):

			mReact = self.reaction[k]

			inc = self.getIndex(mReact.inc)

			for i in range(net.N):

				mNuc = self.nuc[i]

				rate = 0 

				if (mNuc.name in mReact.inc or mNuc.name in mReact.out):

					rate = mReact.getRate(mNuc.name, self.T, self.rho)

				for j in range(mReact.Nin):

					rate *= self.y[inc[j]]

				if (self.rateLimit != None):

					if (rate > self.rateLimit):

						print('limiting rates')

						rate = self.rateLimit 

				R[i] += rate

		return R 

	def buildJ(self):

		# build jacobian 
		J = np.zeros((self.N, self.N))

		for i in range(len(self.reaction)):

			mReact = self.reaction[i]

			inc = net.getIndex(mReact.inc)
			out = net.getIndex(mReact.out)

			ind = inc + out 

			for j in range(len(ind)):

				for k in range(len(inc)):

					y = 1
					for l in range(len(inc)):

						if (inc[l] != inc[k]):

							y *= net.y[k]

					J[ind[j], inc[k]] += y*mReact.getRate(self.name[ind[j]], self.T, self.rho)

		return J 

	def normalize(self):

		tot = np.sum(self.y)

		self.y /= tot

class Reaction:

	def __init__(self, T, rate, inc, out):

		self.T = T
		self.rate = rate 
		self.inc = inc 
		self.out = out 

		self.Nin = len(inc)
		self.Nout = len(out) 

		# interpolate rates, temperatures 
		self.rate_f = interp1d(T, rate)

	def getRate(self, name, T, rho):		

		if (name in self.inc):

			return -self.rate_f(T/1e9)*rho

		else:

			return self.rate_f(T/1e9)*rho

T = 3e9
rho = 1e8
net = Network(T, rho) 

net.updateNuc('he4', 0)
net.updateNuc('c12', 1)
net.updateNuc('o16', 1)
net.updateNuc('ne20', 0)
net.updateNuc('mg24', 0)
net.updateNuc('si28', 0)
net.updateNuc('ni56', 0)

net.buildNetwork()

net.normalize()

net.setRateLimit(1e8)

N = 1000
t = np.logspace(-12, -7, N+1)

I = np.identity(net.N)

allY = np.zeros((net.N, N+1))

allY[:,0] = net.y

for i in range(1, len(t)):

	dt = t[i] - t[i-1]

	J = net.buildJ()
	R = net.buildR()

	A = I - J*dt
	b = np.dot(I - J*dt, net.y) + dt*R 

	Y = np.linalg.solve(A, b)

	net.y = np.copy(Y)

	for j in range(net.N):

		allY[j,i] = Y[j]

print(np.sum(net.y))

for i in range(net.N):

	plt.loglog(t, allY[i,:], label=net.name[i])

plt.legend(loc='best')
plt.show()