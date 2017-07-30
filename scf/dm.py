import numpy as np
from scf_utils import *
import scipy.sparse.linalg as spla
import scipy as sp



def dm(ao_ints, C, nel, scf_params):
    T = ao_ints['T']
    V = ao_ints['V']
    g = ao_ints['g']
    A = ao_ints['A']
    S = ao_ints['S']

    nbasis = scf_params['nbas']
    D = C[:,:nel]
    D = D @ D.T
    maxitr = 20
    for iteration in range(maxitr):

        J, K = get_JK(False,g,D)

        F = T + V + 2 * J - K
        '''
        def E_oper(kappa):
            kappa = kappa.reshape(nbasis,nbasis)
            F_part = -np.einsum("ab,jb->aj",F,kappa)
            F_part += np.einsum("ij,jb->ib",F,kappa)

            g_part = 4*np.einsum("iajb,jb->ia",g,kappa)
            g_part -= np.einsum("ijab,jb->ia",g,kappa)
            g_part -= np.einsum("bjai,jb->ia",g,kappa)

            return (F_part + g_part).ravel()
        E = spla.LinearOperator((nbasis* nbasis, nbasis* nbasis), matvec=E_oper)
        kappa = spla.gmres(E,F.ravel())
        kappa = kappa.reshape(nbasis,nbasis)
        '''
        E = F

        C =  C @ sp.linalg.expm(kappa)
        D = C[:,:nel]
        D = D @ D.T
        
        err = np.sum(F @ D @ S - S @ D @ F)
        print("itr {}, err: {}".format(iteration, err))
        if (err < 10. ** (-5)):
            break


