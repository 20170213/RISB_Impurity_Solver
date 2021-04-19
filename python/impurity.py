#RISB impurity Solver by Pak Ki Henry Tsang, Apr 2021. email: henrytsang222@gmail.com / tsang@magnet.fsu.edu
#
#References: [1] Appendix D of https://doi.org/10.1103/PhysRevX.5.011008  (Phase Diagram and Electronic Structure of Praseodymium and Plutonium)
#            [2] Supplmentary Material of https://doi.org/10.1103/PhysRevLett.118.126401 (Slave Boson Theory of Orbital Differentiation with Crystal Field Effects: Application to UO2)
#
#The function RISB_impurity.iterate outlines how the solver works


from python.matrix import *
from python.diagonalization import *
import numpy as np
from scipy import optimize
from numpy.linalg import inv

class RISB_impurity:

    #############
    # Paramters #
    #############
    
    T=1e-3 #Temperature 
    eta=1e-3 #Broadening
    
    norb=None #Number of orbitals
    snorb=None #Number of orbitals (paramagnetic)
    nom=None #Number of frequency points
    num_eig=6
    verbose=False
    sparse=False
    
    G_FN='G.out'
    R_FN='R.out'
    lambda_FN='lambda.out'
    
    ############
    # Matrices #
    ############
    
    R=None #R
    Rd=None #R^dagger
    l=None #lambda
    Z_matrix=None #R^dagger@R
    identity_matrix=None #identity matrix
    
    Delta_p=None
    D=None
    
    #########
    # Grids #
    #########
    
    omega_grid=None
    omega_matrix_grid=None
    Delta_grid=None
    fermi_grid=None
    fermi_matrix_grid=None
    G_grid=None
    
    ###################
    # Hermitian basis #
    ###################
    
    H_list=None
    tH_list=None
    
    sH_list=None #paramagnetic
    stH_list=None #paramagnetic
    
    ########################################
    # Embedded Hamiltonian sparse matrices #
    ########################################
    
    FH_list=None #Creation/Annihilation operators
    LOBP=None #Local one-body operators
    HYBP=None #hybridizing operators
    HYBPC=None #conjugate of hybridizing operators
    BTHP=None #"bath" operators
    LTBP=None #Local two-body operators
    SPPN=None #Spin operator
    DMTB=None #Density Matrix
    
    ##########################################
    # on-site energies / interaction tensors #
    ##########################################
    
    ob_tensor=None
    tb_tensor=None
    
    def fermi(self,omega,T):
        return 1/(1+np.exp(omega/T))
    
    def denR(self,x):
        return np.power( x*((1.0+0.j)-x), -0.5 )

    def denRm1(self,x):
        return np.power(x*((1.0+0.j)-x),0.5)

    def ddenRm1(self,x):
        return ((0.5+0.j)-x)*np.power(x*((1.0+0.j)-x),-0.5)
    
    def __init__(self,omega_grid,ob_tensor,tb_tensor,norb,T=1e-3,eta=1e-3,G_FN='G.out',R_FN='R.out',lambda_FN='lambda.out',half=False,verbose=False):
        
        #assignments
        self.omega_grid=omega_grid
        self.ob_tensor=ob_tensor
        self.tb_tensor=tb_tensor
        self.T=T
        self.eta=eta
        self.G_FN=G_FN
        self.verbose=verbose
        
        #some constants/constant matrix
        self.norb=norb
        self.snorb=int(norb/2)
        self.grid_size=len(self.omega_grid)
        self.identity_matrix=np.eye(self.norb)
        self.sidentity_matrix=np.eye(self.snorb)
        
        #matrix version of omega grid
        self.omega_matrix_grid=(self.omega_grid[:,np.newaxis]*self.identity_matrix.reshape(self.norb**2)).reshape(self.grid_size,self.norb,self.norb)
        self.somega_matrix_grid=(self.omega_grid[:,np.newaxis]*self.sidentity_matrix.reshape(self.snorb**2)).reshape(self.grid_size,self.snorb,self.snorb)
        
        #calculate fermi function
        self.fermi_grid=self.fermi(self.omega_grid.real,self.T)
        self.fermi_matrix_grid=(self.fermi_grid[:,np.newaxis]*np.ones(self.norb**2)).reshape(self.grid_size,self.norb,self.norb)
        self.sfermi_matrix_grid=(self.fermi_grid[:,np.newaxis]*np.ones(self.snorb**2)).reshape(self.grid_size,self.snorb,self.snorb)
        
        #Hermitian matrix basis
        self.H_list,self.tH_list=Hermitian_list(self.norb)
        self.sH_list,self.stH_list=Hermitian_list(self.snorb)
        
        #####################################################
        # Construct the creation and annihilation operators #
        #####################################################
        
        self.FH_list = build_fermion_op(self.norb)
        
        #################################################################################################
        # Prepares the building blocks of the embedded Hamiltonian operators cdag c, fdag c, fdag f,... #
        #################################################################################################
        
        V2E=self.tb_tensor#two_body_tensor(no,1.0,UT)
        nzV2E = np.nonzero(V2E) #find the non zero entries of two-body tensor
        TBT=(np.transpose(nzV2E),V2E[nzV2E]) #([ijkl,...],[val,...]) of non zero entries two-body tensor
        self.LOBP,self.HYBP,self.HYBPC,self.BTHP,self.LTBP,self.SPPN = build_Hemb_basis(self.norb,self.FH_list,TBT,half=half,spin=False)
        self.DMTB=build_density_matrix_operator(self.FH_list,half=half,spin=False)

        return None
    
    def get_Sigma_grid(self,R,l):

        grid_size=len(self.omega_grid)
        norb=self.norb

        Z_matrix=(R.conj().T)@R #Rdagger@R
        Sigma_grid=-self.omega_grid[:,np.newaxis]*((self.identity_matrix-Z_matrix)@inv(Z_matrix)).reshape(norb**2)\
                    +(inv(R)@l@inv(R.conj().T)).reshape(norb**2) #Eq (D12)
        Sigma_grid.resize(grid_size,norb,norb)
        return Sigma_grid
        
    def get_G_grid(self,Delta_grid,R,l):

        grid_size=len(self.omega_grid)
        norb=self.norb

        self.Sigma_grid=self.get_Sigma_grid(R,l)

        invG_grid=self.omega_matrix_grid-Delta_grid-self.Sigma_grid+1.j*self.identity_matrix*self.eta

        return inv(invG_grid)
    
    def get_Delta_p(self,G_grid,R,l):
        
        norb=self.norb
        grid_size=self.grid_size
        
        #prepare the integrand
        integrand=-(inv(R.conj().T)@G_grid@inv(R)).imag/np.pi*self.fermi_matrix_grid

        #perform trapezoidal integration
        Delta_p=np.trapz(x=self.omega_grid,y=integrand,axis=0)
        return Delta_p
    
    def get_D(self,Delta_grid,G_grid,Delta_p,R,l):
        
        norb=self.norb
        grid_size=self.grid_size
        
        #calculate the left bracket as a matrix function of ( x(1-x) )^(-0.5)
        left=funcmat(Delta_p,self.denR)
        
        #calculate the right bracket : prepare the integrand
        right_integrand=-(Delta_grid@G_grid@inv(R)).imag*self.fermi_matrix_grid/np.pi

        #perform trapezoidal integration
        right=np.trapz(x=self.omega_grid,y=right_integrand,axis=0).T
        return left@right
    
    
    def get_Lambda_c(self,Delta_p, D, R, l ):

        """
        Solves the third RISB equation
        Calculates Lambda_c by using definition of matrix derivative
        """

        #find out number of orbitals
        norb=self.norb
        #decompose Lambda into coefficients
        l_decomposed=inverse_realHcombination(l,self.H_list)
        l_c_decomposed=np.copy(l_decomposed)*0.0

        evals, evecs = np.linalg.eigh(Delta_p)

        derivative=dF(Delta_p,self.tH_list, self.denRm1,self.ddenRm1)
        tt=np.trace(np.matmul(D@(R.T),derivative),axis1=1,axis2=2)

        l_c_decomposed[np.arange(norb**2)]=-l_decomposed[np.arange(norb**2)]-2.0*tt[np.arange(norb**2)].real

        l_c=realHcombination(l_c_decomposed,self.H_list) # Equation 3 (l_c)

        return l_c

    def iterate(self,Delta_grid,R,l,num_eig=2,paramagnetic=True,sparse=False):
        
        self.vprint("R=\n",R)
        self.vprint("l=\n",l)
        
        self.G_grid=self.get_G_grid(Delta_grid,R,l)
        
        self.Delta_p=self.get_Delta_p(self.G_grid,R,l) # Computes eq. D2 and D3 in reference [1], using the rotationally invariant formalism in reference [2]
        
        self.vprint("Delta_p=\n",self.Delta_p)
        
        self.D=self.get_D(Delta_grid,self.G_grid,self.Delta_p,R,l) #Computes eq. D4 in reference [1], using the rotationally invariant formalism in reference [2]
        
        self.vprint("D=\n",self.D)
        
        self.lambda_c=self.get_Lambda_c(self.Delta_p,self.D,R,l) #Computes eq. D5 in reference [1], using the rotationally invariant formalism in reference [2]
        
        self.vprint("lambda_c=\n",self.lambda_c)
        
        # Solve embedding problem
        self.gvec = solve_Hemb(self.D,self.lambda_c, self.ob_tensor, self.tb_tensor, self.LOBP,self.HYBP,self.HYBPC,self.BTHP,self.LTBP,self.SPPN, num_eig=num_eig,sparse=sparse) #Find the ground state of Hamiltonian eq. D9 in reference [1]
        
        #calculate density matrix (note it includes the "duplicated" space)
        norb=self.norb
        norb2=self.norb*2
        denMat = np.zeros((norb2,norb2),dtype=complex)
        for i in range(norb2):
            for j in range(norb2):
                #print(self.DMTB[i*norb2+j])
                denMat[i,j] = self.gvec.conj().T.dot( self.DMTB[i*norb2+j].dot( self.gvec ) )
                
        #This gives occupation
        trDelta_p = np.real(np.trace(self.Delta_p))
        
        #get <f_{a}f_{b}^{\dagger}>
        ffdagger = denMat[norb:,norb:]
        ffdagger = (np.eye(norb,dtype=float) - ffdagger).T
        
        #Compute F1, the matrix corresponding to the first root problem
        #Recall density matrix is in following block structure: [ [c^dagger c, c^daggeer f], [f^dagger c, f^dagger f] ]
        #get_blocks(denMat,no)[2] gets the off-diagonal block c^dagger f (0,1 would be the diagonal elements)
        F1 = ( get_blocks(denMat,norb)[2] - np.dot(R.T, funcmat(self.Delta_p, self.denRm1) ) ) #Computes eq. D7 in reference [1], using the rotationally invariant formalism in reference [2]
        
        #Compute F2, the matrix corresponding to the second root problem
        F2 = ( ffdagger.T - self.Delta_p ) #Computes eq. D8 in reference [1], using the rotationally invariant formalism in reference [2]  
        
        if paramagnetic:
            F1=F1[::2,::2]
            F2=F2[::2,::2]

        #F2 is hermitian
        Fdim = len(np.real(F2))
        F2_real=[]
        F2_imag=[]
        for i in range(Fdim):
            for j in range(i,Fdim):
                F2_real.append(np.real(F2)[i][j])

        for i in range(Fdim):
            for j in range(i+1,Fdim):
                F2_imag.append(np.imag(F2)[i][j])

        #F1 is either all real or all complex
        F1_real=[]
        F1_imag=[]
        for i in range(Fdim):
            for j in range(Fdim):
                F1_real.append(np.real(F1)[i][j])
                F1_imag.append(np.imag(F1)[i][j])

        roots=np.hstack((F2_real,F2_imag,F1_real,F1_imag))
        
        #MAXERR is the maximum value of the many root equations
        print("MAXERR=",max(abs(roots)))
        #self.vprint ("MAXERR=",max(abs(roots))," n=",trDelta_p)

        return roots
    
    def root_problem(self,x,Delta_grid,num_eig=2,paramagnetic=True,sparse=False):
        
        r_decomposed = x[:2*int((self.snorb)**2)] if paramagnetic else x[:2*int((self.norb)**2)]
        l_decomposed = x[2*int((self.snorb)**2):] if paramagnetic else x[2*int((self.norb)**2):]
    
        # build R and Lambda
        R = np.kron(complexHcombination(r_decomposed,self.sH_list),np.eye(2)) if paramagnetic else complexHcombination(r_decomposed,self.H_list)
        l = np.kron(realHcombination(l_decomposed,self.sH_list),np.eye(2)) if paramagnetic else realHcombination(l_decomposed,self.H_list)
        
        return self.iterate(Delta_grid,R,l,num_eig=num_eig,paramagnetic=paramagnetic,sparse=sparse)
    
    def solve(self,Delta_grid,R0,l0,xtol=1e-8,eps=1e-12,factor=1,num_eig=2,paramagnetic=True,sparse=False):
        
        
        #Decompose initial guess
        l0_decomposed = inverse_realHcombination(l0[::2,::2],self.sH_list) if paramagnetic else inverse_realHcombination(l0,self.H_list)  #lambda is hermitian, in general
        r0_decomposed = inverse_complexHcombination(R0[::2,::2],self.sH_list) if paramagnetic else inverse_complexHcombination(R0,self.H_list) #R is not necessarily hermitian
        
        x0 = np.hstack( ( r0_decomposed.real, l0_decomposed.real) )

        #options for the root solver
        options = {'xtol': xtol, #tolerance for convergence
                    'eps': eps, #step size for root finding algorithm
                    'factor': factor #initial factor of the step size (use small to avoid overshooting in the first).
                    }

        rs = optimize.root(self.root_problem,x0,args=(Delta_grid,num_eig,paramagnetic,sparse),method='hybr',options=options)

        print( "RISB root convergence message---------------------------------")
        print( "sucess=",rs.success)
        print( rs.message)
        root_converged = rs.success
        root_value = rs.fun
        
        #get the parameters r, l from the solved root.
        r_decomposed = rs.x[:len(r0_decomposed)]
        l_decomposed = rs.x[len(r0_decomposed):]

        #map r an l to R and Lambda matrix
        R = np.kron(complexHcombination(r_decomposed,self.sH_list),np.eye(2)) if paramagnetic else complexHcombination(r_decomposed,self.H_list)
        l = np.kron(realHcombination(l_decomposed,self.sH_list),np.eye(2)) if paramagnetic else realHcombination(l_decomposed,self.H_list)
        
        output=np.vstack((self.omega_grid,self.G_grid.reshape(self.grid_size,self.norb**2).T)).T
        np.savetxt(self.G_FN,output)
        np.savetxt(self.R_FN,R)
        np.savetxt(self.lambda_FN,l)
        
        return self.G_grid,R,l
    
    def vprint(self,*args,**kwargs):
        if self.verbose:
            print(*args,**kwargs)
