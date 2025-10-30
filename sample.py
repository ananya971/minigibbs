import jax 
import healpy as hp 
import jax.numpy as jnp
import numpy as np
from utils import Cl_to_Dl, load_config
from cg import CG
import matplotlib.pyplot as plt
from plotting import plot_map, plot_c_ells

c = load_config()

def Cl_given_s(alms, nside):
    lmax = 2 * nside
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
    lmax = 2 * nside
    sig_ps = get_sig_ps_healpy(alms)
    np_seed = np.random.randint(0, 2**32-1) # TODO: Make this reproducible!
    key = jax.random.PRNGKey(np_seed)
    subkey = jax.random.split(key, num=lmax+1)
    rho_sq = jnp.zeros(shape = lmax+1)
    for ell in range(0,lmax+1):
        rn = jax.random.normal(subkey[ell], shape = (2*ell-1,)) if ell>0 else jax.random.normal(subkey[ell])
        rn_mod = [np.abs(i)**2 for i in rn] if ell>0 else np.abs(rn)**2
        rho_ell = np.sum(rn_mod) if ell>0 else rn_mod
        rho_sq = rho_sq.at[ell].set(rho_ell)
    C_ell = sig_ps/rho_sq
    C_ell = C_ell.at[:2].set(0.)
    return C_ell

def Cl_given_s_numpy(alms, nside):
    '''Cl function without jax generated random numbers. Stateful.'''
    lmax = 2 * nside
    sig_ps = get_sig_ps_healpy(alms)
    rho_sq = np.zeros(shape = lmax+1)
    for ell in range(0, lmax+1):
        rn = np.random.standard_normal(size = (2*ell-1,)) if ell>0 else np.random.standard_normal()
        rn_mod = [np.abs(i) for i in rn] if ell>0 else np.abs(rn)**2
        rho_ell = np.sum(rn_mod) if ell>0 else rn_mod
        rho_sq[ell] = rho_ell
    C_ell = sig_ps/rho_sq
    C_ell = C_ell.at[:2].set(0.)
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
            


def Cl_given_s_fin(alms, lmax):
    naive_Cl = hp.alm2cl(alms1 = alms, alms2 = None, lmax = lmax)
    naive_Cl[:2] = 0
    sig_ell = (2 * np.arange(0, lmax+1) + 1) * naive_Cl
    key = jax.random.PRNGKey(c['seed'])
    subkey = jax.random.split(key, num=lmax+1)
    rho_ell = jnp.zeros(shape = lmax+1)

    for ell in range(0, lmax+1):
        rho_ell = rho_ell.at[ell].set(jnp.sum(jax.random.normal(subkey[ell], shape = (2*ell-1,))**2)) if ell>0 else rho_ell.at[ell].set(jax.random.normal(subkey[ell]))
        
    C_ell = sig_ell/rho_ell
    return C_ell


def gibbs(iter, init_ps, data, noise, nside):
    ps = init_ps
    smp_C_ell = []
    smp_salm = []
    for i in range(iter):
        signal_alm = CG(c, data, noise, ps)()
        result_pix = hp.alm2map(np.asarray(signal_alm), nside = c['nside'], lmax= 2 * nside)
        plot_map(result_pix*1e6, norm='hist', title=f'{i=}', unit='uK',
                 show=c['show_plots'], fname=f'map_{i}.png')
        C_ell = Cl_given_s_healpy(np.asarray(signal_alm), nside = nside) # init_signal in pixel space
        ps = C_ell 
        plt.figure()
        plot_c_ells(Cl_to_Dl(C_ell), label='D_ell check', logx=True,
                    legend=True, show=c['show_plots'], fname=f'D_ell_{i}.png')
        
        if iter-i <=10:
            smp_C_ell.append(C_ell)
            smp_salm.append(signal_alm)
    
    return smp_C_ell, smp_salm















     
     
     
     
     
        


