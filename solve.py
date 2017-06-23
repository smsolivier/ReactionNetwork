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

		self.A = np.array([]) # store atomic masses 

		self.reaction = [] 

		self.T = T 
		self.rho = rho

		self.rateLimit = None

	def setRateLimit(self, rateLimit):

		self.rateLimit = rateLimit

	def addReaction(self, reaction):

		self.reaction.append(reaction)

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

	def updateNuc(self, name, A, y):
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
			self.A = np.append(self.A, A)

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
		''' build reaction reates vector ''' 

		R = np.zeros(self.N) # store reaction rates for each isotope 

		# loop over reactions 
		for k in range(len(self.reaction)):

			mReact = self.reaction[k]

			inc = self.getIndex(mReact.inc) # indeces of reactants 

			# loop through all nuclides 
			for i in range(net.N):

				mNuc = self.nuc[i]

				rate = 0 

				# if mNuc in reactants or products 
				if (mNuc.name in mReact.inc or mNuc.name in mReact.out):

					# get the rate at given T, rho 
					rate = mReact.getRate(mNuc.name, self.T, self.rho)

				# multiple by Y 
				for j in range(mReact.Nin):

					rate *= self.y[inc[j]]

				# apply limiter 
				if (self.rateLimit != None):

					if (rate > self.rateLimit):

						print('limiting rates')

						rate = self.rateLimit 

				# add rate in 
				R[i] += rate

		return R 

	def buildJ(self):
		''' build jacobian matrix ''' 

		J = np.zeros((self.N, self.N))

		# loop through reactions 
		for i in range(len(self.reaction)):

			mReact = self.reaction[i]

			inc = net.getIndex(mReact.inc) # reactants indices 
			out = net.getIndex(mReact.out) # products indices 

			ind = inc + out # all isotopes involved in reaction 

			# loop through all reactants + products 
			for j in range(len(ind)):

				# reactant Y's are in rates only 
				for k in range(len(inc)):

					# get derivative with respect to inc[k]
					y = 1
					for l in range(len(inc)):

						if (inc[l] != inc[k]):

							y *= net.y[k]

					J[ind[j], inc[k]] += y*mReact.getRate(self.name[ind[j]], self.T, self.rho)

		return J 

	def massFracToMolFrac(self):

		return self.y/self.A

	def molFracToMassFrac(self):

		return self.y*self.A

	def normalize(self):

		mass = np.sum(self.y*self.A)

		self.y /= mass

	def checkConservation(self):

		return np.sum(self.molFracToMassFrac())

class Reaction:

	def __init__(self, T, rate, inc, out):

		self.T = T
		self.rate = rate 
		self.inc = inc 
		self.out = out 

		self.Nin = len(inc)
		self.Nout = len(out) 

		# if only one reactant => rate is decay constant 
		if (self.Nin > 1):

			rate *= rho 

		self.ident = np.zeros(self.Nin)
		for i in range(self.Nin):

			self.ident[i] = self.inc.count(self.inc[i])

		self.outdent = np.zeros(self.Nout)
		for i in range(self.Nout):

			self.outdent[i] = self.out.count(self.out[i])

		# interpolate rates, temperatures 
		self.rate_f = interp1d(T, rate)

		self.Na = 6.022e23

	def getRate(self, name, T, rho):		

		# if a reactant 
		if (name in self.inc):

			return -self.rate_f(T/1e9)

		# if a product 
		else:

			return self.rate_f(T/1e9)/self.ident[0]

T = 3e9
rho = 1e8
net = Network(T, rho) 

# net.updateNuc('1', 2, 1)
# net.updateNuc('2', 1, 1)
# net.updateNuc('3', 1, 0)
# net.updateNuc('4', 1, 0)
net.updateNuc('he4', 4, 0)
net.updateNuc('c12', 12, 1)
net.updateNuc('o16', 16, 1)
net.updateNuc('ne20', 20, 0)
net.updateNuc('mg24', 24, 0)
net.updateNuc('si28', 28, 0)
net.updateNuc('ni56', 56, 0)

net.buildNetwork()
# T = np.logspace(-1, 2, 10)
# rate = np.ones(10)
# net.addReaction(Reaction(T, rate, ['1', '2'], ['3', '4']))
# net.addReaction(Reaction(T, rate/10, ['3', '4'], ['1', '2']))
# net.addReaction(Reaction(T, rate/5, ['1', '1'], ['3', '4']))
# net.addReaction(Reaction(T, rate*rho, ['1'], ['3', '4']))

# net.normalize()

# net.setRateLimit(1e8)

N = 1000
t = np.logspace(-12, -7, N+1)

I = np.identity(net.N)

allY = np.zeros((net.N, N+1))

allY[:,0] = np.copy(net.y)

cons = np.zeros(N+1)

for i in range(1, len(t)):

	dt = t[i] - t[i-1]

	J = net.buildJ()
	R = net.buildR()

	A = I - J*dt
	b = np.dot(I - J*dt, net.y) + dt*R 

	Y = np.linalg.solve(A, b)

	cons[i] = net.checkConservation()

	allY[:,i] = Y

	net.y = np.copy(Y)

	# net.normalize()

	# print(i/N, end='\r')

# maxiter = 100000
# tol = 1e-6

# for i in range(1, len(t)):

# 	dt = t[i] - t[i-1] 

# 	y0 = np.copy(net.y)

# 	for j in range(maxiter):

# 		yold = np.copy(net.y)

# 		ynew = y0 + dt*net.buildR()

# 		net.y = np.copy(ynew)

# 		if (np.linalg.norm(ynew - yold, 2) < tol):

# 			break 

# 		if (j == maxiter - 1):

# 			print('max iter reached')

# 	allY[:,i] = np.copy(net.y)

print(net.checkConservation())

plt.figure()
for i in range(net.N):

	plt.loglog(t, allY[i,:], label=net.name[i])

plt.legend(loc='best')
plt.show()