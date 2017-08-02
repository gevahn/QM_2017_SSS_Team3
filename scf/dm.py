import numpy as np
from scf_utils import *
import scipy.sparse.linalg as spla
import scipy as sp
import os


def eri_ao_to_mo_mp2(C, g, nocc):
    C_occ = C[:,:nocc]
    C_virt = C[:,nocc:]
    g = np.einsum('ab, acde->bcde', C_virt, g)
    g = np.einsum('ab, cdae->cdbe', C_virt, g)
    g = np.einsum('ab, cade->cbde', C_occ, g)
    g = np.einsum('ab, cdea->cdeb', C_occ, g)
    return g

def dm(ao_ints, C, nocc, scf_params):
    T = ao_ints['T']
    V = ao_ints['V']
    g_AO = ao_ints['g4']
    A = ao_ints['A']
    S = ao_ints['S']

    nbasis = scf_params['nbas']
    D = C[:,:nocc]
    D = D @ D.T
    maxitr = 20
    for iteration in range(maxitr):
        J, K = get_JK(False,g_AO,D)

        F = T + V + 2 * J - K

        F = xform_2(F,C)
        g = xform_4(g,C)

        g = eri_ao_to_mo_mp2(C, g_AO, nocc)
        nvirt = nbasis - nocc
        F_oo = F[:nocc,:nocc]
        F_vv = F[nocc:,nocc:]

        f = np.zeros([nvirt,nocc,nvirt,nocc])
        f += -F_oo.reshape(1,nocc,1,nocc) 
        f += F_vv.reshape(nvirt,1,nvirt,1)

        f += 4 * g - np.swapaxes(g,1,2) - np.einsum("bjai",g)
        '''
        def E_oper(kappa):
            kappa = kappa.reshape(nbasis,nbasis)
            kappa = kappa[nocc:,:nocc]
            F_oo = F[:nocc,:nocc]
            F_vv = F[nocc:,nocc:]
            F_part = -np.einsum("ab,bj->aj",F_oo,kappa)
            F_part += np.einsum("ij,bj->ib",F_vv,kappa)

            g_part = 4*np.einsum("abij,bj->ai",g,kappa)
            g_part -= np.einsum("aibj,jb->ia",g,kappa)
            g_part -= np.einsum("bjai,jb->ia",g,kappa)

            return (F_part + g_part).ravel()
        E = spla.LinearOperator((nbasis* nbasis, nbasis* nbasis), matvec=E_oper)
        kappa = spla.gmres(E,F.ravel())
        kappa = kappa.reshape(nbasis,nbasis)
        E = F
        '''
        E = E_f_part + g_part
        kappa = np.linalg.solve(E,F)

        C =  C @ sp.linalg.expm(kappa)
        D = C[:,:nel]
        D = D @ D.T
        
        err = np.sum(F @ D @ S - S @ D @ F)
        print("itr {}, err: {}".format(iteration, err))
        if (err < 10. ** (-5)):
            break


