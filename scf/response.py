import numpy as np
from scf_utils import *
import scipy.sparse.linalg as spla
import scipy as sp
import os

def response(ao_ints, F, C, L_response, R_response, scf_params):
    g = ao_ints['g4']
    nbas = scf_params['nbas']
    nocc = scf_params['nel']
    nvirt = nbas - nocc
    F = xform_2(F,C)
    g = xform_4(g,C)
    L = np.zeros((3,nocc * nvirt))
    R = np.zeros((3,nocc * nvirt))
    for i in range(3):
        L[i,:] = xform_2(2*L_response[i], C)[:nocc,nocc:].ravel()
        R[i,:] = xform_2(2*R_response[i], C)[:nocc,nocc:].ravel()


    E = np.zeros((nbas, nbas, nbas, nbas))
    E += np.einsum('ij,ab->iajb',np.eye(nbas),F)
    E -= np.einsum('ab,ij->iajb',np.eye(nbas),F)

    E += 4*np.einsum('aijb->iajb',g) - np.einsum("jiab->iajb",g) - np.einsum("ajbi->iajb",g)

    E = E[:nocc, nocc:, :nocc, nocc:]
    E = E.reshape(nocc * nvirt, nocc * nvirt)
    E_inv_R = np.linalg.solve(E,R.T)
    return np.dot(L, E_inv_R) 


