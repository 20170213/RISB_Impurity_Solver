#Matrix-related routines (Python 3)
#
#1. Decomposition/Composition of matrices as linear combination of Hermitian matrices
#2. Function of matrix
#3. Loewner matrix derivative
#
#by Pak Ki Henry Tsang, Apr 2021
#email: henrytsang222@gmail.com / tsang@magnet.fsu.edu

import numpy as np

def Hermitian_list(N,real=False):
    if N==1:
        H_list = np.diag(np.array([1.0],dtype=complex))[np.newaxis]
        tH_list = np.diag(np.array([1.0],dtype=complex))[np.newaxis]
    elif N>1:
        #diagonal hermitian basis
        Hd=np.zeros((N,N,N),dtype=complex)
        Hd[np.arange(N),np.arange(N),np.arange(N)]=1.0 #first 1,...,i,...,N entries are H[i,i]=1.0

        #Off-diagonal hermitian basis
        tmp = []
        mesh=[np.meshgrid(np.array([i]),np.arange(i+1,N),indexing='ij') for i in range(N-1)]
        for i in range(N-1):

            Hr=np.zeros((N-1-i,N,N),dtype=complex)
            Hr[np.arange(N-1-i),mesh[i][0],mesh[i][1]]=1.0
            Hr[np.arange(N-1-i),mesh[i][1],mesh[i][0]]=1.0

            Hi=np.zeros((N-1-i,N,N),dtype=complex)
            if real==False:
                Hi[np.arange(N-1-i),mesh[i][0],mesh[i][1]]=1.0j
                Hi[np.arange(N-1-i),mesh[i][1],mesh[i][0]]=-1.0j
            else:
                Hi[np.arange(N-1-i),mesh[i][0],mesh[i][1]]=1.0
                Hi[np.arange(N-1-i),mesh[i][1],mesh[i][0]]=-1.0

            tmp.append(Hr)
            tmp.append(Hi)

        Hod = np.vstack(tmp)/np.sqrt(2)

        H_list = np.concatenate((Hd,Hod))        
        tH_list = np.transpose(H_list,axes=(0,2,1))

    return H_list, tH_list

def realHcombination(x,H_list):
    '''
    Constructs a matrix by linear combination of hermitian matrices with real coefficients x
    The resulting matrix is always a hermitian matrix
    if one use real-valued H_list, this results in a general, real matrix
    '''
    M,N,Q = H_list.shape

    Htmp=np.zeros((M,N,Q),dtype=complex)

    Htmp[np.arange(M)] = np.multiply(x[:,np.newaxis,np.newaxis],H_list)

    return np.sum(Htmp,axis=0)

def inverse_realHcombination(H,H_list):
    '''
    Decomposes a NxN hermitian matrix into N**2 real coefficients x
    if one use real-valued H_list, this results in a general, real matrix
    '''
    M,N,Q = H_list.shape

    x=np.zeros(M,dtype=complex)

    x[np.arange(M)]=np.trace(np.dot(H_list[np.arange(M)],H),axis1=1,axis2=2)

    return np.real(x)

def complexHcombination(v,H_list):
    '''
    Constructs a general matrix by linear combination of hermitian matrices with complex coefficients
    2*N**2 inputs are required, first N**2 elements are real and the last N**2 elements are imaginary
    '''
    M,N,Q = H_list.shape
    twiceM = v.shape[0]

    Ht=np.zeros((twiceM,N,N),dtype=complex)
    Ht[np.arange(M)]= v[np.arange(M),np.newaxis,np.newaxis]*H_list[np.arange(M)]
    Ht[np.arange(M,twiceM)]=v[np.arange(M,twiceM),np.newaxis,np.newaxis]*H_list[np.arange(M)]*1.j

    return np.sum(Ht,axis=0)

def inverse_complexHcombination(H,H_list):
    '''
    Decomposes a general matrix into its 2*N**2 hermitian coefficients
    '''
    M,N,Q = H_list.shape

    x=np.zeros(M,dtype=complex)

    x[np.arange(M)] = np.trace(np.dot(H_list[np.arange(M)],H) ,axis1=1,axis2=2) 

    return np.hstack((np.real(x),np.imag(x)))

def get_blocks(H, s):
    
    N=H.shape[0]

    S=H[0:s,0:s]
    B=H[s:N,s:N]
    V=H[0:s,s:N]
    Vdagger=H[s:N,0:s]

    return S,B,V,Vdagger

def funcmat(H, function):
    """
    This defines the function of a matrix
    """
    assert(H.shape[0] == H.shape[1])

    eigenvalues,U = np.linalg.eigh(H)
    Udagger = U.conj().T

    functioneigenvalues = function(eigenvalues)

    functionH =  np.dot(np.dot(U,np.diag(functioneigenvalues)),Udagger)
    return functionH


def dF(A, H, function, d_function, eta=1e-15):
    
    evals, evecs = np.linalg.eigh(A)
    Hbar = np.matmul(np.conj(evecs).T, np.matmul( H, evecs) ) # transform H to A's basis

    #create Loewner matrix in A's basis
    loewm = np.zeros(evecs.shape,dtype=complex)
    for i in range(loewm.shape[0]):
        for j in range(loewm.shape[1]):
            if evals[i] != evals[j]:
                loewm[i,j]= ( function(evals[i]) - function(evals[j]) )/(evals[i]-evals[j]+eta)
            else: #evals[i]==evals[j] or i==j
                loewm[i,j] = d_function(evals[i]) # derivative(function, evals[i], dx=1e-12)

    # Perform the Schur product in A's basis then transform back to original basis.
    deriv = np.matmul(evecs, np.matmul( loewm*Hbar, np.conj(evecs).T ) )
    return deriv
