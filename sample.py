import jax 
import json 
import healpy as hp 
from jax import random
import jax.numpy as jnp
import numpy as np
from utils import load_config
from functools import partial
from nifty.re import cg 
from cg import CG
import matplotlib.pyplot as plt

#TODO: Add beam and PWF

c = load_config()

def Cl_given_s(alms, nside):
    map = hp.alm2map(alms, pol = False, nside = nside)
    lmax = 2 * nside
    # alms = hp.map2alm(map, lmax = lmax, pol = False)
    alms_full = expand_alms(alms, lmax = lmax)
    sig_ps = get_sig_ps(alms_full)
    
    key = jax.random.PRNGKey(c['seed'])
    subkey = jax.random.split(key, num=2*lmax-1)
    rho_sq = jnp.zeros(shape = lmax+1)
    for ell in range(lmax+1):
        rn = jax.random.normal(subkey[ell], shape = jnp.asarray((2*lmax-1,)))
        rn_mod = [np.abs(i)**2 for i in rn]
        rho_ell = np.sum(rn_mod)
        rho_sq = rho_sq.at[ell].set(rho_ell)
        
    C_ell = sig_ps/rho_sq
    return C_ell

def Cl_given_s_healpy(alms, nside):
    '''Sample from the power spectrum C_ell given the signal alms, and the nside.
    "Healpy compliant", so using only the alms values for m>=0 as healpy returns.'''
    # map = hp.alm2map(alms, pol = False, nside = nside)
    lmax = 2 * nside
    sig_ps = get_sig_ps_healpy(alms)
    key = jax.random.PRNGKey(c['seed'])
    subkey = jax.random.split(key, num=2*lmax-1)
    rho_sq = jnp.zeros(shape = lmax+1)
    for ell in range(lmax+1):
        rn = jax.random.normal(subkey[ell], shape = jnp.asarray((2*lmax-1,)))
        rn_mod = [np.abs(i)**2 for i in rn]
        rho_ell = np.sum(rn_mod)
        rho_sq = rho_sq.at[ell].set(rho_ell)
        
    C_ell = sig_ps/rho_sq
    return C_ell



def expand_alms(alms, lmax):
    '''Expand a set of a_lm values returned by healpy for m>=0, and convert to a full set of alm values of size (lmax+1)**2
    
    Parameters
    ----------
    
    alms : Set of alm values returned by healpy. Must be of shape (lmax+1)*lmax/2 + lmax + 1 for a given lmax
    lmax : lmax for the set of alms, must be consistent with the shape of the alms
    
    Returns:
    ordered : An array of alms of shape (lmax+1)**2
     '''
    # Try with jax thingys
    assert int((lmax * (lmax+1))/2 + lmax + 1) == alms.shape[0], "Shape of alms do not match given lmax"
    ordered = jnp.zeros(shape= (lmax+1)**2, dtype = jnp.complex128)
    for ell in range(lmax+1):
        for m in range(ell, -ell-1, -1): # Iterate over all m vals
                idx = hp.Alm.getidx(lmax, ell, np.abs(m)) # Only valid for +ve m
                diff = alms.shape[0] - (lmax+1) # Difference is basically the ell=0 vals removed
                ordered = ordered.at[idx].set(alms[idx]) if m>=0 else ordered.at[idx+diff].set((-1)**m * np.conjugate(alms[idx])) # Symmetry

    return ordered

def get_sig_ps(alms):
    "Use the full set of alm values of size (lmax+1)**2 to get the signal power spectrum"
    lmax = int(np.sqrt(alms.shape[0])-1)
    sig_ps = jnp.zeros(shape = lmax+1)
    for ell in range(lmax+1):
        sigma_ell = 0
        for m in range(ell, -ell-1, -1):
            idx = hp.Alm.getidx(lmax, ell, np.abs(m)) # Only valid for +ve m
            diff = alms.shape[0] - (lmax+1) 
            sigma_ell += np.abs(alms[idx])**2 if m>=0 else np.abs(alms[idx+diff])**2
        sig_ps = sig_ps.at[ell].set(sigma_ell)
    
    return sig_ps

def get_sig_ps_healpy(alms):
    # Gives me the same result as (2ell+1)*hp.alm2cl
    lmax = hp.Alm.getlmax(alms.shape[0])
    sig_ps = jnp.zeros(shape = lmax + 1)
    for ell in range(lmax+1):
        sigma_ell = 0
        idx = hp.Alm.getidx(lmax, ell, 0)
        sigma_ell += np.abs(alms[idx])**2
        for m in range(1, ell + 1):
            idx = hp.Alm.getidx(lmax, ell, m)
            sigma_ell += 2 * np.abs(alms[idx])**2
        sig_ps = sig_ps.at[ell].set(sigma_ell)
    
    return sig_ps
            

def gibbs(iter, init_ps, data, noise, nside):
    ps = init_ps
    for i in range(iter):
        signal_alm = CG(c, data, noise, ps)()
        C_ell = Cl_given_s_fin(np.asarray(signal_alm), lmax = 2 * nside) # init_signal in pixel space
        ps = C_ell # power spectrum is the initial power spectrum for i = 0, then later is the sampled power spectrum
        # signal_alm_final = hp.almxfl(np.asarray(signal_alm[0]), np.asarray(C_ell[0]))
        result_pix = hp.alm2map(np.asarray(signal_alm), nside = c['nside'], lmax= 2 * c['nside'])
        hp.mollview(result_pix*1e6, norm = 'hist', title = f'{i=}')
        plt.figure()
        plt.plot(jnp.arange(C_ell.shape[0]), C_ell)
        plt.title('C_ell')
        plt.xscale('log')
        plt.show()
    
    return C_ell, signal_alm



def Cl_given_s_fin(alms, lmax):
    naive_Cl = hp.alm2cl(alms1 = alms, alms2 = None, lmax = lmax)
    sig_ell = (2 * np.arange(0, lmax+1) + 1) * naive_Cl
    key = jax.random.PRNGKey(c['seed'])
    subkey = jax.random.split(key, num=lmax+1)
    rho_ell = jnp.zeros(shape = lmax+1)

    for ell in range(1, lmax+1):
        rho_ell = rho_ell.at[ell].set(jnp.sum(jax.random.normal(subkey[ell], shape = (2*ell-1,))**2))
    
    rho_ell = rho_ell.at[0].set(1.)

    # for ell in range(lmax+1):
    #     rn = jax.random.normal(subkey[ell], shape = jnp.asarray((2*lmax-1,)))
    #     rn_mod = [np.abs(i)**2 for i in rn]
    #     rho_ell = np.sum(rn_mod)
    #     rho_sq = rho_sq.at[ell].set(rho_ell)
    # rho_sq = rho_sq.at[0].set(1.)
    # print(f'{rho_sq=}')
    
    C_ell = sig_ell/rho_ell
    return C_ell













     
     
     
     
     
        


