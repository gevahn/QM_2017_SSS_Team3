import numpy as np
import scf_utils
import scipy.sparse.linalg as spla
import scipy as sp



def dm(ao_ints, C, nel):
    T = ao_ints['T']
    V = ao_ints['V']
    g = ao_ints['g']
    A = ao_ints['A']
    S = ao_ints['S']

    nbasis = scf_params['nbas']
    D = C[:,:nel]
    D = D @ D.T

    for iteration in range(maxitr):

        J, K = getJK(g,D)

        F = T + V + 2 * J - K

        def E_oper(kappa):
            F_part = -np.einsum("ab,jb->aj",F,kappa)
            F_part += np.einsum("ij,jb->ib",F,kappa)

            g_part = 4*np.einsum("iajb,jb->ia",g,kappa)
            g_part -= np.einsum("ijab,jb->ia",g,kappa)
            g_part -= np.einsum("bjai,jb->ia",g,kappa)

            return F_part + g_part
        E = LinearOperator((nbasis, nbasis, nbasis, nbasis), matvec=E_oper)
        kappa = spla.gmres(E,F)

        C =  C @ sp.linalg.expm(kappa)
        D = C[:,:nel]
        D = D @ D.T


