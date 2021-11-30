import math
from scipy.optimize import linprog
import gurobipy as gp
import numpy as np
import sys
from config import *
from fractions import Fraction
import sympy as sp
from sympy.solvers.solveset import linsolve
import pickle

verbose = False
optMethod = 'interior-point'

env = gp.Env(empty = True)
env.setParam("DualReductions",0)
env.setParam("OutputFlag",0)
env.start()

maxi = 0

class Polyhydron:
	def __init__(self, A = np.array([[]]), b = np.array([]), A_eq = np.array([[]]), b_eq = np.array([])):
		self.num_ine = b.shape[0]
		self.num_equ = b_eq.shape[0]
		self.gModel = gp.Model(env = env)
		# self.equalities = []
		# self.inequalities = []

		self.dim = 0
		if self.num_ine>0: self.dim = A.shape[1]
		if self.num_equ>0: self.dim = A_eq.shape[1]

		if self.dim>0:
			self.x = self.gModel.addMVar((self.dim),lb = -math.inf)

		if self.num_ine>0: self.gModel.addConstr(A @ self.x >= b)
		if self.num_equ>0: self.gModel.addConstr(A_eq @ self.x== b_eq)

		self.isEmpty = False

	# coef = c, d
	# add an inequality c*x>=d to the Polyhydron
	def add_ine(self,coef):
		c,d = coef
		self.num_ine += 1
		if self.dim == 0:
			self.dim = c.shape[0]
			self.x = self.gModel.addMVar((self.dim),lb = -math.inf)
		# self.inequalities.append((c, d))
		self.gModel.addConstr(c @ self.x >= d)

	def add_equ(self,coef):
		c,d = coef
		self.num_ine += 1
		if self.dim == 0:
			self.dim = c.shape[0]
			self.x = self.gModel.addMVar((self.dim),lb = -math.inf)
		# self.equalities.append((c, d))
		self.gModel.addConstr(c @ self.x ==  d)

# input: a Polyhydron
# output: whether the LP is feasible, if yes, a solution x
def feasibility(P:Polyhydron):
	if P.isEmpty: 
		return False,np.array([])
	if P.num_equ==0 and P.num_ine==0:
		return True,0
	P.gModel.update()
	P.gModel.optimize()
	if P.gModel.Status==3:
		var_value = np.array([])
	else:
		var_value = P.x.X
	P.isEmpty = P.gModel.Status==3
	return P.gModel.Status!=3, var_value

# def findRational(P:Polyhydron):
# 	# find the rational number version of this 
# 	rationalLP = gp.Model(env = env)
# 	x = rationalLP.addMVar((P.dim,), lb=-math.inf, vtype = 'I')
# 	q = rationalLP.addMVar((1,),vtype='I')
# 	constrList = P.gModel.getConstrs()
# 	constr_index= {c: i for i, c in enumerate(constrList)}

# 	matrix = []

# 	for constr in constrList:
# 		if constr.slack == 0:
# 			row = P.gModel.getA().toarray()[constr_index[constr]]

# 			matrix.append(row)

# 			rhs = constr.rhs
# 			rationalLP.addConstr(row @ x == rhs + rhs * q)
# 		# else:
# 		# 	if constr.sense == '<':
# 		# 		rationalLP.addConstr(row @ x <= rhs + rhs * q)
# 		# 	if constr.sense == '>':
# 		# 		rationalLP.addConstr(row @ x >= rhs + rhs * q)
# 	matrix = np.array(matrix)


# 	rationalLP.optimize()
# 	global maxi
# 	maxi = max(np.amax(np.array(x.X)),maxi)
# 	print("maximum intege so far",maxi)
# 	result = np.array(x.X).astype(int).astype('object')
# 	denom = Fraction(1,int(q.X[0])+1)
# 	result = result * denom
# 	print("making one constraint strict")
# 	return result

def findRational(P:Polyhydron):
	# find the rational number version of this 
	constrList = P.gModel.getConstrs()
	constr_index= {c: i for i, c in enumerate(constrList)}

	matrix = []
	b = []

	for constr in constrList:
		if constr.slack == 0:
			row = P.gModel.getA().toarray()[constr_index[constr]].astype('int')
			matrix.append(row)
			b.append(int(constr.rhs))
	matrix = sp.Matrix(np.array(matrix))
	b = sp.Matrix(np.array(b))

	x = sp.symbols('x0:%d'%P.dim)
	solution = linsolve((matrix,b), x)

	f = solution.args[0]
	freeSym = f.free_symbols
	if len(freeSym)>0:
		print("has",len(freeSym),"free symbols!")
	result = np.array(f.subs([(i,0) for i in freeSym]))
	print("making one constraint strict")
	# print(result)
	return result

# input: a polyhydron P and a list constraints that needs to be strict
# output: if feasible, output a point in P that is strict on the required constraints
def interiorPoint(P:Polyhydron, constrs):
	if not feasibility(P)[0]:
		return None
	points = np.array([P.x.X.copy()])
	if rational_solution:
		rationalPoints = [findRational(P)]
	onBoundary = True
	while onBoundary:
		point = np.average(points, axis = 0)
		onBoundary = False
		for constr in constrs:
			c = constr[0]
			d = constr[1][0]
			slack = np.inner(c,point) - d
			# print(slack)
			if abs(slack)<epsilon:
				# constrs.index(constr)
				# print("try to make constraint", constrs.index(constr),"strict, slack", slack)
				onBoundary = True
				P.gModel.addConstr(c @ P.x <= d+2)
				P.gModel.setObjective(-c @ P.x)
				P.gModel.update()
				P.gModel.optimize()
				if rational_solution:
					rationalPoints.append(findRational(P))
				points = np.concatenate((points, P.x.X.copy().reshape(1,-1)), axis = 0)
				break
	print("\nthe result is an averange of",points.shape[0],"points\n")
	# print(rationalPoints)
	if rational_solution:
		rationalPoint = np.full((P.dim,),0)
		for a in rationalPoints:
			rationalPoint = rationalPoint + a
		rationalPoint /= len(rationalPoints)
		return rationalPoint
	else:
		return point



#input: same as feasibility
#output: solves Ax>=b, min obj*x 
#        in the form of (opt success, opt value)
def lp(P:Polyhydron,obj):
	if P.num_equ==0 and P.num_ine==0:
		return -math.inf
	P.gModel.setObjective(np.array(obj) @ P.x)
	P.gModel.update()
	P.gModel.optimize()
	if P.gModel.Status==5:
		fun = -math.inf
	else:
		if P.gModel.Status == 3:
			fun = math.inf
		else:
			fun = P.gModel.getObjective().getValue()
	return P.gModel.Status!=3, fun

# def affineHull(P:Polyhydron):
# 	if not feasibility(P):
# 		return None
# 	for i in range(P.num_ine):
# 		c,d = P.A[i], P.b[i]
# 		success, fun, x = lp(P,c)
# 		if success and abs(fun-d)<1e-6:
# 			P.add_equ((c,np.array([d])))
# 			P.del_ine(i)

# 	return P.A_eq, P.b_eq

#decides whether affine subspace D1 is a subset of D2
def subset(D1, D2):
	A1,b1 = D1
	A2,b2 = D2
	if len(b2)==0:
		return True
	else:
		if len(b1)==0: return False

	non_empty = np.linalg.norm(A1.dot(np.linalg.pinv(A1)).dot(b1)-b1)
	if non_empty>1e-5: return False

	vertex_r = A2.dot(np.linalg.pinv(A1)).dot(b1)-b2
	null_space_r = A1.T.dot(np.linalg.pinv(A1.T)).dot(A2.T)-A2.T
	if np.linalg.norm(vertex_r)<1e-5 and np.linalg.norm(null_space_r)<1e-5:
		return True
	else:
		return False

def polyMul(a1, a2):
	return np.convolve(a1,a2)

def evaluatePoly(poly, x):
	res = 0
	for i in range(poly.shape[0]):
		res += poly[i] * x**i
	return res

def test():
	A = np.array([[1,1],
		 	      [-1,-1]])
	b = np.array([1,-1])
	A = np.array([
		[-1,-1],
		[1,0],
		[0,1],
		[-4,-1]])
	b = np.array([-1,0,0,-2])
	P=Polyhydron(A=A, b=b)
	# P.add_equ((np.array([1,1]),np.array([2])))
	P.add_ine((np.array([1,1]),np.array([0])))
	# print(affineHull(P))
	print(feasibility(P))
	print(interiorPoint(P, [(A[i],b[i].reshape(1,-1)) for i in range(4)]))

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	test()