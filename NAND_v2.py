import numpy as np
from polyLP_v5 import *
from util_v4 import *
from config import *

np.set_printoptions(threshold=sys.maxsize)

# change a number n to base b
def numberToBase(n, b, l):
    if n == 0:
        return np.array([0]*l)
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    digits = digits+[0]*(l-len(digits))
    return np.array(digits[::-1])

#transition probability from 2 nodes to 1 node, represent u by 2
def transProb(t):
	prob = {
		(0,0,0):[0,0,1],  (0,0,1):[1,0,-1], (0,0,2):[0,0,0],
		(0,1,0):[0,1,-1], (0,1,1):[1,-1,1], (0,1,2):[0,0,0],
		(0,2,0):[0,0,1],  (0,2,1):[1,-1,1], (0,2,2):[0,1,-2],
		(1,0,0):[0,1,-1], (1,0,1):[1,-1,1], (1,0,2):[0,0,0],
		(1,1,0):[1,-2,1], (1,1,1):[0,2,-1], (1,1,2):[0,0,0],
		(1,2,0):[0,1,-1], (1,2,1):[0,2,-1], (1,2,2):[1,-3,2],
		(2,0,0):[0,0,1],  (2,0,1):[1,-1,1], (2,0,2):[0,1,-2],
		(2,1,0):[0,1,-1], (2,1,1):[0,2,-1], (2,1,2):[1,-3,2],
		(2,2,0):[0,0,1],  (2,2,1):[0,2,-1], (2,2,2):[1,-2,0]
	}
	return np.array(prob[t])

# extend a constant matrix to the field of polynomials by assigning higher order coefficients to be 0
# (shape) changes to (shape,degree+1)
def extendToPoly(M,degree):
	sh = M.shape
	comp = np.full(sh+(degree,),0)
	M = M.reshape(sh+(1,))
	M = np.concatenate((M,comp),axis = len(sh))
	return M

def generatePolyLP(r):
	# first generate Bin, Bout and B, 3^(r) * 3^(r+1)
	Bin = np.full((3**r,3**(r+1)),0)
	Bout = np.full((3**r,3**(r+1)),0)
	for i in range(3**r):
		for j in range(3**(r+1)):
			if (numberToBase(i,3,r)==numberToBase(j,3,r+1)[1:]).all():
				Bin[i,j] = 1
			if (numberToBase(i,3,r)==numberToBase(j,3,r+1)[:-1]).all():
				Bout[i,j] = 1
	B=Bout-Bin
	if verbose:
		print(B)

	# generate P, 3^r * 3^(r-1)
	P = np.kron(np.identity(3**(r-1)),np.array([[1,1,1]]).T)
	if verbose:
		print("P\n",P)

	# generate C
	degree = 2*(r-1)
	C = np.full((3**r,3**(r-1),degree+1),0)
	C[:,:,0] = 1
	for i in range(3**r):
		for j in range(3**(r-1)):
			old_string = numberToBase(i,3,r)
			new_string = numberToBase(j,3,r-1)
			for k in range(r-1):
				C[i,j] = polyMul(C[i,j],transProb((old_string[k],old_string[k+1],new_string[k])))[:degree+1]
	if verbose:
		print("C\n",C)

	#generate psi 3^r
	psi = np.full((3**r),0)
	for i in range(3**r):
		if numberToBase(i,3,r)[0] ==2:
			psi[i]=1
	if verbose:
		print("psi\n",psi)

	#generate matrix A and vector xi
	Aupper = np.full((3**(r+1),3**(r-1),degree+1),0)
	Alower = np.full((3**(r+1),3**(r-1),degree+1),0)
	for i in range(degree+1):
		Aupper[:,:,i] = Bout.T.dot((extendToPoly(P,degree)-C)[:,:,i])
		Alower[:,:,i] = Bout.T.dot(extendToPoly(P,degree)[:,:,i])
	Btrans = extendToPoly(B.T,degree)
	Zeros = np.full((3**(r+1),3**r,degree+1),0)
	Aupper = np.concatenate((Aupper,Btrans,Zeros),axis = 1)
	Alower = np.concatenate((Alower,Zeros,Btrans),axis = 1)
	A = np.concatenate((Aupper,Alower),axis = 0)

	xi = extendToPoly(Bout.T.dot(psi),degree)
	xi = np.concatenate((np.full((3**(r+1),degree+1),0),xi), axis = 0)
	if verbose:
		print("A\n",A)
		print("xi\n",xi)


	# add the condition that non-u-only values should be zero
	u_only = np.full((2**(r-1),3**(r-1)+2*3**r,degree+1),0)
	j=0
	for i in range(3**(r-1)):
		str = numberToBase(i,3,r-1)
		if not 2 in str: 
			u_only[j][i][0] = 1
			j+=1
	A_eq = u_only
	b_eq = np.full((2**(r-1),degree+1),0)
	if verbose:
		print("A_eq")
		print(A_eq)

	return A,xi,A_eq,b_eq


def test():
	print(numberToBase(5,3,3))
	print(numberToBase(0,3,3))
	print(extendToPoly(np.array([2,3]),3))
	np.set_printoptions(threshold=sys.maxsize)
	generatePolyLP(3)

# test()
r = 4
A, xi, A_eq, b_eq = generatePolyLP(r)
print("finish generation")
divergeRange = degreeBound
for div in range(degreeBound+1):
	P=PolyLP(A,xi,A_eq,b_eq, new_degree = degreeBound)
	print("tring diverge level",div,"degree bound",degreeBound)
	P.setDiverge(div)
	success, x, finalLP = P.subspaceElimination()
	print("searching result:",success,"at diverge level",div)
	save_object(P,"result.plh")
	if success:
		print("index of the final LP",finalLP)
		print("printing the coefficients of solution-------------------------------------")
		sol = P.SE2coef(x)
		print("coefficients of p")
		print(sol[0])
		print("coefficients of q")
		print(sol[1])

		for j in range(A.shape[0]):
			re = - polyMul(sol[1],xi[j])
			for i in range(A.shape[1]):
				# print(re)
				# print(polyMul(A[j,i],sol[0][i]))
				re += np.concatenate(( polyMul(A[j,i],sol[0][i]),[0]*(div+1)))
			print("result on constraint",j,":",re)
			resultWrong = False
			for k in range(re.shape[0]):
				if re[k]>epsilon:
					break
				if re[k]<-epsilon:
					resultWrong = True
					break
			if resultWrong: break

		if resultWrong:
			print("The result is not correct!")
		else:
			print("Checking finished, result correct")
		np.savetxt('coefficients.csv',sol[0],delimiter=',')
		break

print("-----------------------\n the maximal delta that works:",P.rangeCalculation())
