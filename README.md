# RISB_Impurity_Solver

References: [1] Appendix D of https://doi.org/10.1103/PhysRevX.5.011008  (Phase Diagram and Electronic Structure of Praseodymium and Plutonium)
            [2] Supplmentary Material of https://doi.org/10.1103/PhysRevLett.118.126401 (Slave Boson Theory of Orbital Differentiation with Crystal Field Effects: Application to UO2)
            
This is an numerical solver for the Anderson Impurity Model using the Rotationally Invariant Slave Bosons (RISB) approximation. This code is tested under python3.6.9 (intel) with numpy version 1.17.0 and scipy version 1.3.1. It will NOT run properly on python 2.

One should first prepare files, default to be 'one_body.inp', 'two_body.inp', 'R.inp', 'lambda.inp', 'Delta.inp' for the solver, otherwise it would not work properly. The instructions to format these files can be found in this document.

usage: solver.py [-h] --mu MU --norb NORB [--eta ETA] [--T T] [--xtol XTOL]
                 [--eps EPS] [--neig NEIG] [--obody OBODY] [--tbody TBODY]
                 [--Gout GOUT] [--Rout ROUT] [--lout LOUT] [--Delta DELTA]
                 [--Rinp RINP] [--linp LINP] [--half] [--paramagnetic]
                 [--sparse] [--verbose]

optional arguments:
  -h, --help      show this help message and exit
  --mu MU         chemical potential
  --norb NORB     number of orbitals
  --eta ETA       broadening
  --T T           Temperature
  --xtol XTOL     xtol
  --eps EPS       eps
  --neig NEIG     number of eigenvalues to find
  --obody OBODY   one-body energy file (input)
  --tbody TBODY   two-body energy file (input)
  --Gout GOUT     Green's function output file
  --Rout ROUT     R output file
  --lout LOUT     lambda output file
  --Delta DELTA   Hybridization Function file (input)
  --Rinp RINP     R output file
  --linp LINP     lambda output file
  --half          halves the embedded Hamiltonian Hilbert space (warning: just
                  leave this alone)
  --paramagnetic  seek paramagnetic solution
  --sparse        use sparse eigenvalue solver (faster)
  --verbose       verbose output


Some additional notes:
- xtol is the tolerance of the root finder, the root finder might not converge if this value is set to too small
- eps is the step size of the root finder, the root finder might not converge if this value is set to too large or too small 
- T or temperature will affect the smearing of the fermi-function. A larger value helps convergence, but one should refrain from increasing this too much to avoid superficial behaviors as RISB (so-far) is a 0-temperature formalism.
- the default file for Hybridization function is Delta.inp
- the default file for initial Guess of R,lambda are respectively R.inp and lambda.inp
- the option --half will decrease the size of the Hilbert space of the Hamiltonian to diagonalize, which speeds up calculation in the paramagnetic case without causing problems in the single-band case. But it is not clear if this would break calculations in other cases, so the best is to leave this option alone (which is default to be turned off).
- the option --paramagnetic will force the solver to seek paramagnetic solutions, one should always use this option if looking for paramagnetic solutions.
- the option --sparse will force the solver to use sparse eigenvalue routines, which will speed up calculations greatly especially if there are many orbitals


Instruction to prepare files for the solver:

  say, we have an impurity

    a. with two correlated sites, which including spin we have 4 orbitals. Site A has energy -1 and site B has energy 0, intersite hopping strength is 0.5 and correlation strenght U_A=2.0 and U_B=3.0
    b. chemical potential of 0
    c. seeking paramagnetic solution

  then the following flags should be supplied to the solver

    --norb=4
    --mu=0
    --paramagnetic

  and the necessary files to prepare are

    1. 'one_body.inp', which stores the on-site energies of different orbitals

        0 0 -1.0
        1 1 -1.0
        2 2 0.0
        3 3 0.0
        0 2 0.5
        2 0 0.5
        1 3 0.5
        3 1 0.5
      
      0 0 -1.0 means in the Hamiltonian it is written as -1.0*c^dagger_0*c_0
      First four lines are the on-site energies. orbital 0 and 1 are the spin orbitals of site A, which has energy -1, and orbital 2 and 3 are the spin orbitals of site B. Note that we count orbitals from 0.
      
      The last four lines are the inter-site hopping strength, note that spin up can only hop to spin up, and so on. 
    
    2. 'two_body.inp', which stores the correlation energies
    
        0 0 1 1 1.0
        1 1 0 0 1.0
        2 2 3 3 1.5
        3 3 2 2 1.5
      
      0 0 1 1 1.0 means in the Hamiltonian we have 1.0*c^dagger_0*c_0*c^dagger_1*c_1
      We have U_A=2.0 and U_B=3.0, the correlation energies here are exactly half of of U_A and U_B to avoid double-counting.

    3. 'R.inp'
    
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
      
      We have 4 orbitals, so we need 4*4=16 values for the initial guess of the R-matrix. As you can see, this combination is exactly the row-major representation of the identity matrix
      
    4. 'lambda.inp'
    
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        
     We have 4 orbitals, so we need 4*4=16 values for the initial guess of the lambda-matrix. 
     
     5. 'Delta.inp' , which specifies the hybridization function as well as the frequency grid.
     
      Say we want to define our grid, linearly spaced, from frequency omega=-1.0 to omega=1.0, then say I have a made-up bath (not of any good) that is a identity matrix at every frequency, we can have Delta.inp written as
      
        -1.0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        -0.8 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        -0.6 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        -0.4 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        -0.2 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        0.0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        0.2 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        0.4 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        0.6 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        0.8 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        1.0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
        
      We can see that the first column is the frequency, and the rest are the row-major representation of the hybridization function. Note that this bath is completely made-up and only serve the purpose of demonstration. In actual use you'll want to use a denser, wider grid.
