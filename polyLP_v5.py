import numpy as np
from util_v4 import *
from multiprocessing import Pool, cpu_count, Value, Array
from config  import *
import pickle


# define synchronized global variables
# P is defined here cause it cannot be pickled
P = Polyhydron()

constraint_index = []
add_constr = []
finish_constr = []
for i in range(500):
	constraint_index.append(Value('i',1))
	add_constr.append(Value('b',False))
	finish_constr.append(Value('b',False))

# k denote the degree of the resulting polynomial
# i denote the index of constraint
def getCoef(poly,i,k,A,b):
	coef_p = np.full((poly.dim_org, poly.new_degree+1), 0)
	coef_q = np.full((poly.new_degree+1), 0)
	coef_nonhomo = 0

	coef_p[:,max(0,k-poly.degree):min(k,poly.new_degree)+1] = A[i,:,::-1][:,poly.degree+max(0,k-poly.degree)-k:poly.degree+min(k,poly.new_degree)-k+1]
	s = k-poly.diverge_level-1
	if s>=0:
		coef_q[max(0,s-poly.degree):min(s,poly.new_degree)+1] = -b[i,::-1][poly.degree+max(0,s-poly.degree)-s:poly.degree+min(s,poly.new_degree)-s+1]
	s = k-poly.diverge_level
	if s>=0 and s<=poly.degree:
		coef_nonhomo = poly.xi[i,s]

	if verbose:
		print("constraint",i,k)
		print(coef_p)
		print(coef_q)
		print(coef_nonhomo)

	c = np.concatenate( (coef_p.reshape(-1), coef_q.reshape(-1)) )
	d = np.array([coef_nonhomo])

	return c,d

def tryEliminate(poly, i):
	if constraint_index[i].value!=-1:
		affCoef = getCoef(poly, i, constraint_index[i].value-1, poly.A,poly.xi)
		coef = getCoef(poly, i,constraint_index[i].value, poly.A,poly.xi)
		minimum = lp(P,affCoef[0])[1]
		maximum = -lp(P,-affCoef[0])[1]

		# debug info
		if verbose:
			print("dealing constraint",i,"degree",constraint_index[i])
			print("coefficient\n",affCoef,"\n",coef)
			print("maximum",maximum)
			print("minimum",minimum)

		if abs(minimum-affCoef[1][0])<epsilon and abs(maximum-affCoef[1][0])<epsilon: # if subset(affP, affCoef):
			add_constr[i].value = True

			if progress and i%10==0: 
				print("removing one subspace at index", i, constraint_index[i].value)

			constraint_index[i].value +=1
			if constraint_index[i].value > poly.tot_degree:
				finish_constr[i].value = True
				if progress: 
					print("finish at index", i)

		if affCoef[1][0] < minimum-epsilon or affCoef[1][0]>maximum+epsilon:
			finish_constr[i].value = True

			if progress: 
				print("finish at index", i)


#A: num_constraints * dim * (degree+1)
#xi: num_constraints * (degree+1)
# inequalities: A * p -xi * delta^{c+1} * q >= xi * delta^c 
# equalities: A_eq * p - b_eq * delta^{c+1} * q = b_eq * delta^c 
class PolyLP:

	def __init__(self, A, xi, A_eq=np.array([]), b_eq=np.array([]), diverge_level = 0, new_degree = -1):
		self.degree = A.shape[2]-1
		self.num_constraints = A.shape[0]
		self.dim_org = A.shape[1]
		self.num_equ = A_eq.shape[0]
		self.A = A
		self.xi = xi
		self.diverge_level = diverge_level
		self.A_eq = A_eq
		self.b_eq = b_eq
		if new_degree == -1:
			new_degree = self.dim_org*self.degree
		self.new_degree = new_degree
		self.tot_degree = new_degree + self.degree
		global P
		P = Polyhydron()

	def setDiverge(self,di):
		self.diverge_level = di

	def subspaceElimination(self):
		#add all equality constraints
		for i in range(self.num_equ):
			if progress:
				print("initializing equality",i)
			for k in range(self.tot_degree+1):
				P.add_equ(getCoef(self,i,k,self.A_eq,self.b_eq))

		# add inequalities of constant coefficients
		for i in range(self.num_constraints):
			if progress:
				print("initializing inequality",i)
			P.add_ine(getCoef(self,i,0,self.A, self.xi))

		# forcing coefficients to get a better looking result---------------------------

		# harmonic = np.array([0,0,2,0,0,4,-2,1,2,0,0,2,0,0,3,0,1,2,0,0,2,1,0,3,0,1,2])
		# left = np.full((27, (self.dim_org+1)*(self.new_degree+1), ),0)
		# for i in range(27):
		# 	left[i,i*(self.new_degree+1)]=1
		# newLeft = np.full((1, (self.dim_org+1)*(self.new_degree+1), ), 0)
		# newLeft[0,14*(self.new_degree+1)+1]=1
		# left = np.concatenate((left,newLeft),axis = 0)
		# harmonic = np.concatenate((harmonic,np.array([-1])))
		# P.gModel.addConstr(left @ P.x == harmonic)
		
		coef_hol = np.array([0,0,0,0,0,2,0,2,2,0,0,0,0,0,1,1,2,2,1,1,1,2,1,2,1,2,2])
		left = np.full((27, (self.dim_org+1)*(self.new_degree+1), ),0)
		for i in range(27):
			left[i,i*(self.new_degree+1)]=1
		newLeft = np.full((13, (self.dim_org+1)*(self.new_degree+1), ), 0)
		newLeft[0,11*(self.new_degree+1)+1]=1
		newLeft[1,14*(self.new_degree+1)+1]=1
		newLeft[2,6*(self.new_degree+1)+1]=1
		newLeft[3,5*(self.new_degree+1)+1]=1
		newLeft[4,7*(self.new_degree+1)+1]=1
		newLeft[5,8*(self.new_degree+1)+1]=1
		newLeft[6,15*(self.new_degree+1)+1]=1
		newLeft[7,16*(self.new_degree+1)+1]=1
		newLeft[8,26*(self.new_degree+1)+1]=1
		newLeft[9,23*(self.new_degree+1)+1]=1
		newLeft[10,24*(self.new_degree+1)+1]=1
		newLeft[11,20*(self.new_degree+1)+1]=1
		newLeft[12,21*(self.new_degree+1)+1]=1
		left = np.concatenate((left,newLeft),axis = 0)
		coef_hol = np.concatenate((coef_hol,np.array([1,1,4,3,2,2,-1,0,-1,-1,-4,2,-4])))
		P.gModel.addConstr(left @ P.x == coef_hol)

		for i in range(self.num_constraints):
			constraint_index[i].value = 1
		has_update = True
		finalLP = {}
		for i in range(self.num_constraints):
			finalLP[i] = 0

		if progress:
			print("finish initializing Polyhydron")

		# do subspace elimination
		while has_update and feasibility(P)[0]:

			has_update = False
			for i in range(self.num_constraints):
				add_constr[i].value = False
				finish_constr[i].value = False

			# find if we can add constraint at constraint i of degree constraint_index[i]
			pool = Pool(self.num_constraints)
			pool.starmap(tryEliminate, [(self,i) for i in range(self.num_constraints)])
			pool.close()
			pool.join()

			for i in range(self.num_constraints):
				if add_constr[i].value:
					affCoef = getCoef(self, i, constraint_index[i].value-2, self.A, self.xi)
					coef = getCoef(self, i, constraint_index[i].value-1, self.A, self.xi)
					P.add_ine(coef)
					P.add_equ(affCoef)
					has_update = True

					finalLP[i] = constraint_index[i].value-1
				if finish_constr[i].value:
					constraint_index[i].value = -1

		self.success, self.x = feasibility(P)
		P.gModel.write('finalLP.lp')

		if self.success:
			aliveConstrs = []
			for i in range(self.num_constraints):
				if finalLP[i]<self.tot_degree: aliveConstrs.append(getCoef(self, i, finalLP[i], self.A, self.xi))
			with open("aliveConstrs.txt","wb") as fp:
				pickle.dump(aliveConstrs,fp)
			self.x = interiorPoint(P, aliveConstrs)

		return self.success, self.x, finalLP


	def SE2coef(self, x):
		sep = self.dim_org*(self.new_degree+1)
		coef_p = x[:sep].reshape((self.dim_org, self.new_degree+1))
		coef_q = x[sep:]
		move = np.array([0]*self.diverge_level+[1])
		self.coef_p = coef_p
		self.coef_q = np.concatenate((move,coef_q))
		return self.coef_p, self.coef_q

	def rangeCalculation(self):
		upperBound = 1
		lowerBound = 1e-5
		resultPoly = []
		for j in range(self.num_constraints):
			re = - polyMul(self.coef_q,self.xi[j])
			for i in range(self.A.shape[1]):
				re += np.concatenate(( polyMul(self.A[j,i],self.coef_p[i]),[0]*(self.diverge_level+1)))
			resultPoly.append(re)
		# use binary search to find the range that all polynomials are at least 0
		while abs(upperBound-lowerBound)>1e-6:
			delta = (upperBound+lowerBound)/2
			work = True
			for i in range(self.num_constraints):
				if evaluatePoly(resultPoly[i], delta)<0:
					work = False
					break
			if work:
				lowerBound = delta
			else:
				upperBound = delta
		return delta

def test():
	AList = []
	xiList = []
	A = np.array([[[1,1],[2,-1]],[[-1,-2],[-2,-1]]])
	xi = np.array([[3,1],[-3,0]])
	AList.append(A)
	xiList.append(xi)
	AList.append( np.array([[[1,1],[2,-1]],[[-1,-1],[-2,1]]]))
	xiList.append( np.array([[3,0],[-3,0]]))
	AList.append( np.array([[[1,1],[2,-1]],[[2,1],[2,-5]]]))
	xiList.append( np.array([[3,0],[4,-1]]))

	for i in range(len(AList)):
		print("\ntry example",i)
		A=AList[i]
		xi=xiList[i]
		Po = PolyLP(A, xi)
		f, x, fLP= Po.subspaceElimination()
		print("search success:",f)
		if f:
			print(x)
			sol = Po.SE2coef(x)
			print(sol)
			for j in range(A.shape[0]):
				re = - polyMul(sol[1],xi[j])
				for i in range(A.shape[1]):
					re += np.concatenate(( polyMul(A[j][i],sol[0][i]),[0]))
				print("result on constraint",j,":",re)

if __name__ == "__main__":
	test()