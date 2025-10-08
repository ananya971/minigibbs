import jax 
import json 
import healpy as hp 
from jax import random
import jax.numpy as jnp
import numpy as np
from utils import Cl_to_Dl, Dl_to_Cl, load_config
from nifty.re import cg 
from cg import CG
import matplotlib.pyplot as plt
import camb
from plotting import plot_map, plot_c_ells

c = load_config()

pars = camb.set_params(H0=67.72,
            ombh2=0.022,
            omch2=0.1192,
            mnu=0.06,
            omk=0,
            tau=0.054,
            As=2.09e-9,
            ns=0.9667,
            halofit_version="mead",
            lmax=2 * c['nside'],
        )
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit = 'K', lmax=2 * c['nside'])
power_spec = powers["total"]
TTDl = power_spec[:,0]


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
    key = jax.random.PRNGKey(c['seed'])
    subkey = jax.random.split(key, num=lmax+1)
    rho_sq = jnp.zeros(shape = lmax+1)
    for ell in range(0,lmax+1):
        rn = jax.random.normal(subkey[ell], shape = (2*ell-1,)) if ell>0 else jax.random.normal(subkey[ell])
        rn_mod = [np.abs(i)**2 for i in rn] if ell>0 else np.abs(rn)**2
        rho_ell = np.sum(rn_mod) if ell>0 else rn_mod
        rho_sq = rho_sq.at[ell].set(rho_ell)
    C_ell = sig_ps/rho_sq
    plt.figure()
    plt.plot(jnp.arange(sig_ps.shape[0]), sig_ps, label = 'sig_ps')
    plt.plot(jnp.arange(rho_sq.shape[0]), rho_sq, label = 'rho_sq')
    plt.plot(jnp.arange(C_ell.shape[0]), C_ell, label = 'C_ell')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return C_ell

def Cl_given_s_numpy(alms, nside):
    '''Cl function without jax generated random numbers. So statefull.'''
    lmax = 2 * nside
    sig_ps = get_sig_ps_healpy(alms)
    rho_sq = np.zeros(shape = lmax+1)
    for ell in range(0, lmax+1):
        rn = np.random.standard_normal(size = (2*ell-1,)) if ell>0 else np.random.standard_normal()
        rn_mod = [np.abs(i) for i in rn] if ell>0 else np.abs(rn)**2
        rho_ell = np.sum(rn_mod) if ell>0 else rn_mod
        rho_sq[ell] = rho_ell
    C_ell = sig_ps/rho_sq
#    D_ell = Cl_to_Dl(C_ell)

#    plt.figure()

#    plt.plot(np.arange(C_ell.shape[0]), D_ell)
#    plt.xscale('log')
#    plt.savefig(f'cells_{it}.png')
#    plt.close()

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
    for i in range(iter):
        signal_alm = CG(c, data, noise, ps)()
        result_pix = hp.alm2map(np.asarray(signal_alm), nside = c['nside'], lmax= 2 * nside)
        plot_map(result_pix*1e6, norm='hist', title=f'{i=}', unit='uK',
                 show=c['show_plots'], fname=f'map_{i}.png')
#        hp.mollview(result_pix*1e6, norm = 'hist', title = f'{i=}', unit = 'uK', )
#        plt.savefig(f'map_{i}.png')
        C_ell = Cl_given_s_numpy(np.asarray(signal_alm), nside = nside) # init_signal in pixel space
        ps = C_ell 
        # print(f'{ps=}')# power spectrum is the initial power spectrum for i = 0, then later is the sampled power spectrum
        plt.figure()
        plot_c_ells(Cl_to_Dl(C_ell), label='D_ell check', logx=True,
                    legend=True, show=c['show_plots'], fname=f'D_ell_{i}.png')
#        plt.plot(jnp.arange(C_ell.shape[0]), Cl_to_Dl(C_ell), label = 'D_ell check')
#        # plt.plot(jnp.arange(C_ell.shape[0]), Cl_to_Dl(ps), label = 'ps check')
#        plt.legend()
#        plt.xscale('log')
#        plt.show()
        # plt.figure()
        # plt.plot(jnp.arange(C_ell.shape[0]), Cl_to_Dl(C_ell), label = 'D_ell')
        # plt.plot(np.arange(C_ell.shape[0]), TTDl, label = 'TTDl')
        # plt.legend()
        # plt.title(f'D_ell iteration {i}')
        # plt.xscale('log')
        # plt.show()
    
    return C_ell, signal_alm















     
     
     
     
     
        


