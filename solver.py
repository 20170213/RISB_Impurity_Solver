#!/usr/bin/env python3
#RISB impurity Solver by Pak Ki Henry Tsang, Apr 2021. email: henrytsang222@gmail.com / tsang@magnet.fsu.edu

#####################
# arguments parsing #
#####################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--U", help="Hubbard U", required=True, type=float)
parser.add_argument("--mu", help="chemical potential", required=True, type=float)
parser.add_argument("--norb", help="number of orbitals", required=True, type=int)
parser.add_argument("--eta", help="broadening", default=5e-3, type=float)
parser.add_argument("--T", help="Temperature", default=1e-3, type=float)
parser.add_argument("--xtol", help="xtol", default=1e-5, type=float)
parser.add_argument("--eps", help="eps", default=1e-8, type=float)
parser.add_argument("--neig", help="number of eigenvalues to find", default=2, type=int)
parser.add_argument("--obody", help="one-body energy file (input)",default='one_body.inp')
parser.add_argument("--tbody", help="two-body energy file (input)",default='two_body.inp')
parser.add_argument("--Gout", help="Green's function output file",default='G.out')
parser.add_argument("--Rout", help="R output file",default='R.out')
parser.add_argument("--lout", help="lambda output file",default='lambda.out')
parser.add_argument("--Delta", help="Hybridization Function file (input)",default="Delta.inp")
parser.add_argument("--Rinp", help="R output file",default='R.inp')
parser.add_argument("--linp", help="lambda output file",default='lambda.inp')
parser.add_argument("--verbose", help="verbose output",action='store_true')
args = parser.parse_args()

norb=args.norb
num_eig=args.neig
verbose=args.verbose
  
T=args.T
eta=args.eta
mu=args.mu
U=args.U

Delta_FN=args.Delta
one_body_FN=args.obody
two_body_FN=args.tbody
G_FN=args.Gout
R_FN=args.Rout
lambda_FN=args.lout
Ri_FN=args.Rinp
lambdai_FN=args.linp

xtol=args.xtol
eps=args.eps
    
######################
# Begin main program #
######################

import numpy as np
from python.impurity import *
    
#We want to first read in hybridization function for many orbitals, it should be arranged in the following manner
#omega reDelta00 imDelta00 reDelta01 imDelta01 

Delta_input=np.loadtxt(Delta_FN,dtype=complex)

grid_size=len(Delta_input)

Delta_grid=[]
omega_grid=[]
for row in Delta_input:
    omega_grid.append(row[0]) #First element in row is the frequency
    Delta_grid.append(row[1:]) #The remainder is hybridization function
omega_grid=np.array(omega_grid) #Cast as numpy array
Delta_grid=np.array(Delta_grid).reshape(grid_size,norb,norb) #Cast as numpy array


#Next is to read in on-site and inter-site energies, it should look like
# 0 0 0.0    #e0
# 1 1 -1.0    #e1
# ...

#Next is to read in two-body terms
# 0 0 1 1 1.0    #U01
# 1 1 0 0 1.0    #U10
# ...

ob_tensor = np.zeros((norb,norb),dtype=float) #initialize tensor
for line in open(one_body_FN):
    if line.startswith('#'):
        continue
    ls=line.split()
    idx1=int(ls[0])
    idx2=int(ls[1])
    en=float(ls[2])
    ob_tensor[idx1,idx2]=en
for idx in range(norb):
    ob_tensor[idx,idx]-=mu

tb_tensor = np.zeros((norb,norb,norb,norb),dtype=float) #initialize tensor
for line in open(two_body_FN):
    if line.startswith('#'):
        continue
    ls=line.split()
    idx1=int(ls[0])
    idx2=int(ls[1])
    idx3=int(ls[2])
    idx4=int(ls[3])
    en=float(ls[4])
    tb_tensor[idx1,idx2,idx3,idx4]=en   #up up down down


#Next is to read in the guesses of R-matrix and lambda-matrix. 

#Read in R
R0=np.zeros(norb**2,dtype=complex)
for line in open(Ri_FN):
    if line.startswith('#'):
        continue
    ls=line.split()
    length=len(ls)
    assert(length==norb**2)
    for idx in range(length):
        R0[idx]=complex(ls[idx])
R0.resize(norb,norb)

l0=np.zeros(norb**2,dtype=complex)
for line in open(lambdai_FN):
    if line.startswith('#'):
        continue
    ls=line.split()
    length=len(ls)
    assert(length==norb**2)
    for idx in range(length):
        l0[idx]=complex(ls[idx])
l0.resize(norb,norb)

#########################
# Initialize the solver #
#########################


RISB_solver=RISB_impurity(omega_grid,ob_tensor,tb_tensor,norb,T=T,eta=eta,G_FN=G_FN,R_FN=R_FN,lambda_FN=lambda_FN,verbose=verbose)

RISB_solver.solve(Delta_grid,R0,l0,xtol=xtol,eps=eps,factor=1,sparse=True)
