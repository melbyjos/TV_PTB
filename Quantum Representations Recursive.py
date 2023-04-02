#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time as time
import matplotlib.pyplot as plt
import math
import cmath
import gmpy2 as gp
from gmpy2 import mpc

#####################################################################################

#Here are some test arrays of level sets that will come in handy:
test1 = [311]
test  = [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91]
test1 = [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95,99,103,107,111,131,151,171,191,211,231,251,271,291,311,351,391,431,471,511,611,711,811,911,1011]
test111 = [3,5,7,9,11,13,15,17,19,23,29,35,41,51,61,71,81]
test1 = [3,5,7,9,11,13,15,17,19,21,23,25,29,31,35,41,51]
test1 = [11,13,15,17,19,21,23,25,31,41,51,61,71,81,91,101,111]
test1 = [11,13,15,17,19,21,23,25,31,41,51,61,71,81,91,101,111,201,301,401,501]

#Chen-Yang Test set for the figure eight knot complement:
CYtest = [11,13,15,17,19,21,23,25,31,41,51,61,71,81,91,101,111,201,301,401,501,701,1001]

##################################################################################################

#This function provides a list of the admissible colors for the given level set r:
def colors(r):
    I = [i for i in range((r-3)//2+1)]
    return I

#Primitive root of unity for which we are computing all of our invariants, as a function of the level set r:
def A(r):
    B = cmath.exp(cmath.pi*1j/(r))
    return B
################################################################################################

################################################################
#Supporting functions for quantum integers and factorials
################################################################

def Q(n,A):
    z = (-A)**(n) - (-A)**(-n)
    return z.imag*1j

def Qp(n,A):
    z = (-A)**(n) + (-A)**(-n)
    return z.real
def f1(N,A):
    if N==0:
        return 1
    elif N>0:
        return np.prod([Q(i,A) for i in range(1,N+1)])
    else: 
        return 0
def f1p(N,A):
    if N==0:
        return 2
    elif N>0:
        return np.prod([Qp(i,A) for i in range(1,N+1)])
    else: 
        return 0
def f2(N,A):
    if N==0:
        return 1
    else:
        temp=1
        for i in range(N,0,-2):
            temp=temp*Q(i,A)
            continue
        return temp
def mu(n,A):
    return (-A)**(n*(n+2))
def lamb(n,A):
    return -Q(2*n+2,A)

################################################################################################

################################################################
#Functions used to construct the representations for C_R and C_L
################################################################
def R(c,n,m,A):
    x = f1(m,A)*f2(2*c+2*n+1,A)*f1p(2*c+n+1,A)
    y = f1(n,A)*f2(2*c+2*m+1,A)*f1p(2*c+m+1,A)
    return x/y

def Mminus(N,n,m,A):
    x = Q(m,A)*((-A)**(-2*N+2*m))
    y = Q(n+1,A)
    return x/y
def Mplus(N,n,m,A):
    x = ((-A)**(-2*N+2*m+2))*Q(-2*N+2*m+2,A)*Qp(-2*N+m+1,A)
    y = Q(n+1,A)
    return x/y
def Mmid1(c,N,n,m,A):
    x = (A-A**(-1))*(lamb(c+m,A))**(2)+A*(Q(m+1,A)**(2))*R(c,m+1,m,A)-A**(-1)*(Q(m,A)**2)*R(c,m,m-1,A)-Q(2,A)*lamb(c+n,A)
    y = Q(2,A)*Q(n+1,A)
    return x/y

def Mmid(c,N,n,m,A):
    x = (A-A**(-1))*(Qp(2*N-2*m-1,A))**(2)+A*(Q(m+1,A)**(2))*R(c,m+1,m,A)-A**(-1)*(Q(m,A)**2)*R(c,m,m-1,A)+Q(2,A)*Qp(2*N-2*n-1,A)
    y = Q(2,A)*Q(n+1,A)
    return x/y

def makeM(N,c,n,A):
    bigM = np.zeros(shape=(N,N),dtype=complex)
    bigM[0,0] = Mmid(c,N,n,0,A)
    bigM[0,1] = Mplus(N,n,0,A)
    bigM[N-1,N-2] = Mminus(N,n,N-1,A)
    bigM[N-1,N-1] = Mmid(c,N,n,N-1,A)
    for m in range(1,N-1):
        bigM[m,m] = Mmid(c,N,n,m,A)
        bigM[m,m-1] = Mminus(N,n,m,A)
        bigM[m,m+1] = Mplus(N,n,m,A)
    return bigM

################################################################################################

################################################################
#Build libraries for R and L representations
################################################################

def makeRepR(N,c,A):
    S = np.zeros(shape=(N,N),dtype = complex)
    S[0,0] = 1
    for n in range(1,N):
        M = makeM(N,c,n-1,A)
        S[:,n] = M.dot(S[:,n-1])
    return S

def libR(array):
    lib = {}
    #np.empty(dtype = object)
    for r in array:
        ts = time.process_time()
        I = colors(r)
        repsr = [makeRepR((r-1)//2-c,c,A(r)) for c in I]
        lib[r] = repsr
        print('Time for R reps at r = %s: %s seconds' %(r,time.process_time()-ts))
    return lib



time_start = time.process_time()

Rlib = libR(test)

print('-------------------------------------------------------------------------------------')

time_elapsed = (time.process_time() - time_start)
print('Time to build R Library: %s seconds' % time_elapsed)

print('-------------------------------------------------------------------------------------')


#Once we have the R rep library, we can construct the corresponding library of L reps using these two functions:

def makeRepL(N,r,c,A):
    repR = Rlib[r][c]
    #S = (b_{n,m})
    S = np.linalg.inv(np.array([[repR[m,n]/R(c,n,m,A) for m in range(N)] for n in range(N)]))
    return S

def libL(array):
    lib = {}
    for r in array:
        ts = time.process_time()
        I = colors(r)
        repsr = [makeRepL((r-1)//2-c,r,c,A(r)) for c in I]
        lib[r] = repsr
        print('Time for L reps at r = %s: %s seconds' %(r,time.process_time()-ts))
    return lib


time_start = time.process_time()

Llib = libL(test)

print('-------------------------------------------------------------------------------------')

time_elapsed = (time.process_time() - time_start)
print('Time to build L Library: %s seconds' % time_elapsed)

print('-------------------------------------------------------------------------------------')


################################################################################################

#Define representations of C_R and C_L:

def genRep(r,c,string):
    #r >= 3 is odd TV parameter
    #c is color from list of colors, N = (r-1)/2 - c
    #k is choice of primitive 2r root of unity 
    #string is the string of L's and R's 
    N = (r-1)//2-c
    if 'R' in string:
        repR = Rlib[r][c]
    if 'L' in string:
        repL = Llib[r][c]
    rho = np.zeros(shape=(N,N), dtype=complex)
    first = string[0]
    if first == 'L':
        rho = repL
    else: 
        rho = repR
    for char in string[1:]:
        if char == 'L':
            rho = rho.dot(repL)
        else:
            rho = rho.dot(repR)
    return rho

def cc(r):
    if r % 4 == 3:
        return (r-3)//4
    
def midRep(r):
    return (2*np.pi)/(r)*np.log(abs(np.trace(genRep(r,cc(r),'RL')))**2)


x = np.zeros(shape = (len(test),))
y = np.zeros(shape = (len(test),))
i=0
print('Middle color values:')
for r in test:
    if r%4==3:
        x[i] = r
        #print(x)
        y[i] = midRep(r)
        print((r,midRep(r)))
        i=i+1
plt.plot(x,y)

print('-------------------------------------------------------------------------------------')

        

################################################################################################

################################################################
#Computing the Turaev-Viro
################################################################

def TV(r,string):
    I = colors(r)
    traces = [abs(np.trace(genRep(r,c,string)))**2 for c in I]
    TV = sum(traces)
    return TV

#This function computes the logarithm of TV_r, as the Volume Conjecture requires
def QV(r,string):
    return ((2*np.pi)/(r-2))*np.log(TV(r,string))

################################################################################################

################################################################
#Given a list of level sets (subset of the representations library created above) and a list of monodromies 
#(as strings in R,L), this function constructs a library of QV_r values keyed by the string representing the monodromy
################################

def QVLib(array,strings):
    lib = {}
    time_start = time.process_time()
    for stg in strings:
        QVallr = [[r,QV(r,stg)] for r in array]
        lib[stg] = QVallr
    time_elapsed = (time.process_time() - time_start)
    print('Time to compute QV Library: %s seconds' % time_elapsed)
    print('-------------------------------------------------------------------------------------')
    print('Library:')
    print(lib)
    return lib


################################################################################################

QVLib(test,['RL'])


# In[13]:


def AllStrings(n, arr, i):  
  
    if i == n: 
        printTheArray(arr, n)  
        return
      
    # First assign "0" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 0
    generateAllBinaryStrings(n, arr, i + 1)  
  
    # And then assign "1" at ith position  
    # and try for all other permutations  
    # for remaining positions  
    arr[i] = 1
    generateAllBinaryStrings(n, arr, i + 1)

