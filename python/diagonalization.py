import numpy as np
from scipy.special import comb
from scipy.sparse import csr_matrix

def build_fermion_op(no):
    """
    This builds the creation operator in the occupation ordered basis for "no" amount of orbitals
    Ordered basis is <n>-ordered, while unordered basis is binary (numerical) ordered
    Blocks are ordered from <n>=0,<n>=1,...,<n>=no
    For the half-fill block, i.e. <n>=no/2 block, it is block-ordered from high to low spin
    Final results are stored in a (2**no) sized square sparse matrix of type np.int (np.int32)
    The intermediate results are of mixed data types in order to speed up the code and conserve memory
    """

    FH_list=[] #list of creation operators (starts empty)
    no2=no*2
    hsize = 2**no2 #size of Hilbert Space (full size)

    #decimal representation of basis (unordered)
    dec_array=np.arange(hsize,dtype=np.uint32)
    #create the binary-ordered basis
    binary_array = np.mod(np.divide(dec_array[:,np.newaxis],np.power(2,np.arange(no2,dtype=np.uint32))),2).astype(np.uint8)
    #find occupation of each basis
    sum_array = np.sum(binary_array,axis=1,dtype=np.uint8) 
    #find the position of unordered basis in occupation ordering
    occupation_basis_id = np.argsort(sum_array).astype(np.uint32)
    #print occupation_basis_id
    #create ordered basis, this allows us to find position of binary vector from position of ordered basis
    occupation_basis_array = np.zeros((hsize,no2),dtype = np.uint8)
    occupation_basis_array[np.arange(hsize)] = binary_array[occupation_basis_id]
    #split the sorted basis into 3 parts, isolating the part of half-fill in particular
    half_size = comb(no2,no,exact=True)
    start= sum([comb(no2,i,exact=True) for i in range(no)])
    end= sum([comb(no2,i,exact=True) for i in range(no+1)])
    half_fill_array = occupation_basis_array[start:end]
    front_array = occupation_basis_array[:start]
    after_array = occupation_basis_array[end:]
    half_fill_id = occupation_basis_id[start:end]
    front_id = occupation_basis_id[:start]
    after_id = occupation_basis_id[end:]
    #alternating 0.5, -0.5, 0.5, -0.5 ,...
    spin_value=(np.power(-1,np.arange(no2))+np.ones(no2))/4-(np.power(-1,np.arange(no2)+1)+np.ones(no2))/4
    #get total spin of the half-fill array
    spin_array=np.sum(half_fill_array*spin_value,axis=1)
    spin_sort_id=np.argsort(spin_array)
    spin_basis_array = np.zeros(half_fill_array.shape,dtype = np.uint8)
    spin_basis_array[np.arange(half_size)] = half_fill_array[spin_sort_id]
    sorted_basis_array=np.concatenate((front_array,spin_basis_array,after_array))
    spin_basis_id = np.zeros(half_size,dtype = np.uint32)
    spin_basis_id[np.arange(half_size)] = half_fill_id[spin_sort_id]
    sorted_basis_id=np.concatenate((front_id,spin_basis_id,after_id))
    #find the position of ordered basis when not ordered, this allows us to find position of ordered basis vector
    #from position of binary basis
    #binary_basis_id= np.zeros(hsize,dtype=np.uint32)
    #binary_basis_id[occupation_basis_id]=np.arange(hsize,dtype=np.uint32) 
    binary_basis_id= np.zeros(hsize,dtype=np.uint32)
    binary_basis_id[sorted_basis_id]=np.arange(hsize,dtype=np.uint32) 

    for orb in range(no2):
        #The criteria is simple, a zero in any of |B> means it is part of column
        col_pos=np.where(occupation_basis_array[:,orb]==0)[0].astype(np.uint32) #position of column

        #Get the binary representation of the <A| vectors
        a_vec = np.zeros((int(hsize/2),no2),dtype=np.uint8)
        a_vec[np.arange(int(hsize/2))]=occupation_basis_array[col_pos]
        a_vec[:,orb]=1
        #find the position of the binary vectors, then find their n-ordered position
        pos_a_vec = np.sum(np.multiply(a_vec,np.power(2,np.arange(no2,dtype=np.uint32))),axis=1)
        row_pos = binary_basis_id[pos_a_vec]

        #Find the exponent due to anticommutation relations
        expo=np.zeros(int(hsize/2),dtype=np.int64)
        expo=np.sum(occupation_basis_array[col_pos,0:orb+1],axis=1,dtype=np.int64)

        #Get the number <A|cdag[orb]|B>
        data=np.zeros((int(hsize/2)),dtype=np.int64)
        data=np.power(-1,expo,dtype=np.int64)

        #Create the sparse matrix from position of <A|, position of |B> and <A|cdag[orb]|B>
        FH = csr_matrix((data, (row_pos, col_pos)), shape=(hsize, hsize), dtype=np.int64)
        FH_list.append(FH)

    return FH_list

def build_Hemb_basis(no,FH_list,V2E,half=True,spin=False):
    """
    LOBP: 		local one-body part 	--- cdagger[i].c[j]
    HYBP,HYBPC:	hybridization part 	--- cdagger[i].f[j] , fdagger[j].c[i]
    BTHP: 		bath part 		--- f[j].fdagger[i]
    LTBP:		local two-body part	--- cdagger[i].c[j].cdagger[k].c[l]
    Stored as np.complex128 sparse matrices
    """

    no2 = no*2

    if half==True:
        if spin == False:        
            start= sum([comb(no2,i,exact=True) for i in range(no)])
            end= sum([comb(no2,i,exact=True) for i in range(no+1)])
            hsize = end-start#comb(no2,no,exact=True)
        else: #obtain the spin-0 sector
            n=np.ones(no+1,dtype=np.uint32)*no
            k=np.arange(no+1)
            spin_pos=np.insert(np.cumsum(comb(n,k,exact=False).astype(np.int)**2),0,0)
            start = sum([comb(no2,i,exact=True) for i in range(no)])+spin_pos[int(spin_pos.shape[0]/2)-1]
            end = sum([comb(no2,i,exact=True) for i in range(no)])+spin_pos[int(spin_pos.shape[0]/2)]
            hsize = end-start#comb(no2,no,exact=True)
    else:
        hsize= 2**no2
        start=0
        end=hsize

    imesh,jmesh=np.meshgrid(np.arange(no),np.arange(no),indexing='ij')
    imesh = list(imesh.flatten())
    jmesh = list(jmesh.flatten())

    LOBP = [FH_list[i].dot(FH_list[j].getH())[start:end,start:end].astype(complex) for i,j, in zip(imesh,jmesh)]
    HYBP = [FH_list[j].dot( FH_list[i+no].getH() )[start:end,start:end].astype(complex)  for i,j, in zip(imesh,jmesh)]
    HYBPC= [FH_list[i+no].dot( FH_list[j].getH() )[start:end,start:end].astype(complex) for i,j, in zip(imesh,jmesh)  ]
    BTHP = [(FH_list[j+no].getH()).dot( FH_list[i+no] )[start:end,start:end].astype(complex)  for i,j, in zip(imesh,jmesh) ]

    LTBP = [FH_list[el[0]].dot((FH_list[el[1]].getH()).dot(FH_list[el[2]].dot((FH_list[el[3]].getH()))))[start:end,start:end].astype(complex)\
            for el in V2E[0]]

    #build spin operators:
    #build total spin operators S+
    Sp = csr_matrix((hsize,hsize), dtype=np.int32)
    for i in range(no):
        Sp += FH_list[2*i].dot(FH_list[2*i+1].getH())[start:end,start:end]
    #build total spin operators S-
    Sm = csr_matrix((hsize,hsize), dtype=np.int32)
    for i in range(no):
        Sm += FH_list[2*i+1].dot(FH_list[2*i].getH())[start:end,start:end]
    Sz = csr_matrix((hsize,hsize), dtype=np.int32)
    for i in range(no):
        Sz += ( 0.5*FH_list[2*i].dot(FH_list[2*i].getH()) - 0.5*FH_list[2*i+1].dot(FH_list[2*i+1].getH()) )[start:end,start:end]
    #build S^2 operator
    SPPN = Sm.dot(Sp)+Sz.dot(Sz)+Sz
    SPPN = SPPN.tocsr().astype(complex)

    return LOBP,HYBP,HYBPC,BTHP,LTBP,SPPN

def build_density_matrix_operator(FH_list,half=True,spin=False):
    """
    DMTB: cdag[i]c[j] and fdag[i]f[j]
    Stored as complex sparse matrices
    """
    no2 = len(FH_list)
    no = int(no2/2)

    if half==True:
        #hsize = comb(no2,no,exact=True)
        if spin == False:        
            start= sum([comb(no2,i,exact=True) for i in range(no)])
            end= sum([comb(no2,i,exact=True) for i in range(no+1)])
            hsize = end-start#comb(no2,no,exact=True)
        else: #obtain the spin-0 sector
            n=np.ones(no+1,dtype=int)*no
            k=np.arange(no+1)
            spin_pos=np.insert(np.cumsum(comb(n,k,exact=False).astype(int)**2),0,0)
            start = sum([comb(no2,i,exact=True) for i in range(no)])+spin_pos[int(spin_pos.shape[0]/2)-1]
            end = sum([comb(no2,i,exact=True) for i in range(no)])+spin_pos[int(spin_pos.shape[0]/2)]
            hsize = end-start#comb(no2,no,exact=True)
    else:
        hsize= int(2**no2)
        start=0
        end=hsize

    DMTB = []
    for i in range(no2):
        for j in range(no2):
                DMTB.append( FH_list[i].dot( FH_list[j].getH())[start:end,start:end].astype(complex) )

    return DMTB

def solve_Hemb(D,lambda_c, H1E, V2E, LOBP,HYBP,HYBPC,BTHP,LTBP,SPPN, num_eig=6, spin_pen=0, sparse=False, verbose=False):
#def solve_Hemb(D, H1E, LAMBDA, V2E, spin_pen, num_eig, LOBP,HYBP,HYBPC,BTHP,LTBP,SPPN, verbose=0):

    from scipy.sparse.linalg import eigsh
    #from scipy.linalg import eigh as seigh

    """
    Solve embedding Hamiltonian and return the ground state wavefunction and eigenvalue
    Eigenvalue problem solved by scipy.linalg.eigsh (or use Primme.eigsh, if needed)
    """

    #reshape the input matrices (flatten for list comprehension loop) 
    nod = len(D)
    D = D.reshape(nod**2)
    Dc = D.conj()
    LAMBDA = lambda_c.reshape(nod**2)
    H1E = H1E.reshape(nod**2)

    nzV2E = np.nonzero(V2E) #find the non zero entries of two-body tensor
    TBT=(np.transpose(nzV2E),V2E[nzV2E]) #([ijkl,...],[val,...]) of non zero entries two-body tensor
    Hemb = sum(H1E*LOBP)+sum(D*HYBP)+sum(Dc*HYBPC)+sum(LAMBDA*BTHP)
    if len(TBT[0])!=0:
        Hemb +=sum(TBT[1]*LTBP)

    #Solve for least eigenvalues
    if not(sparse):
        vals, vecs = np.linalg.eigh(Hemb.todense())
    else:
        vals, vecs = eigsh(Hemb, k=num_eig, which='SA')
    #vals, vecs = eigsh(Hemb)

    #get ground state wavefunction
    gvec = vecs[:,0]

    if verbose ==1:
        print ('eigenvalues=',vals)
        #print 'eigenvector=',gvec


    return gvec