import nifty.re as jft
import jax.numpy as jnp 
import numpy as np 
import healpy as hp 
import jax
from utils import load_config
from nifty.re import cg

c= load_config()

key = jax.random.PRNGKey(c['seed'])

class CG:
    def __init__(self, c, data, noise, C_ell):
        self.c = c
        self.data = data
        self.noise = noise
        self.Ninv = np.asarray(noise)**-1
        self.C_ell = C_ell
        self.npix = 12 * self.c['nside']**2
        self.ps = self.C_ell.shape[0]
        self.res = hp.nside2resol(nside = self.c['nside'], arcmin = True)/60 #[deg]
    
    def apply_A(self, s, n_freq):
        s_beam = hp.almxfl(alm = s, fl = hp.gauss_beam(fwhm = self.res, lmax = 2 * self.c['nside']))
        spix_beam = hp.alm2map(s_beam, nside = self.c['nside'], lmax = 2 * self.c['nside'])
        return np.tile(spix_beam, n_freq)
    
    def apply_A_transpose(self, stacked, n_freq):
        s_summed = stacked.reshape(n_freq, self.npix).sum(axis = 0)
        spix_summed = hp.map2alm(s_summed, lmax = 2 * self.c['nside'])
        s_beamed = hp.almxfl(alm = spix_summed, fl = hp.gauss_beam(fwhm = self.res, lmax = 2 * self.c['nside']))
        return s_beamed
    

    def apply_mat(self,x):
        '''Apply matrix, finish doc'''
        x = np.asarray(x)
        alm_cl = hp.almxfl(alm = x, fl = np.sqrt(self.C_ell)) # S^(1/2)x
        A_pix_alm = self.apply_A(alm_cl, self.c['nfreqs'])
        NA_pix = np.asarray(self.Ninv) * A_pix_alm
        ATNA_pix = self.apply_A_transpose(NA_pix, self.c['nfreqs'])
        CATNA_alm = hp.almxfl(ATNA_pix, fl = np.sqrt(self.C_ell))
        return CATNA_alm + x

    def rhs(self):
        term_1a = self.Ninv * self.data
        term_1b = self.apply_A_transpose(term_1a, self.c['nfreqs'])
        term_1 = hp.almxfl(term_1b, fl = np.sqrt(self.C_ell))

        hp.mollview(hp.alm2map(term_1*1e6, nside = self.c['nside'], lmax = 2 * self.c['nside']), norm = 'hist', title = 'term_1')

        omega1 = jft.random_like(key, jax.ShapeDtypeStruct(shape = self.noise.shape, dtype = jnp.float64))

        term_2a = np.sqrt(self.Ninv) * np.asarray(omega1)
        term_2b = self.apply_A_transpose(term_2a, self.c['nfreqs'])
        term_2 = hp.almxfl(term_2b, fl = np.sqrt(self.C_ell))

        hp.mollview(hp.alm2map(term_2*1e6, nside = self.c['nside'], lmax = 2 * self.c['nside']), norm = 'hist', title = 'term_2')

        omega0 = jft.random_like(key, jax.ShapeDtypeStruct(shape = (int((self.ps * (self.ps -1))/2 + (self.ps)) ,), dtype = jnp.complex64))

        return term_1 + term_2 + np.asarray(omega0)
    
    def apply_cg(self):
        rhs = self.rhs()
        init_sig = hp.synalm(self.C_ell, lmax = 2 * self.c['nside'])
        return cg(self.apply_mat, rhs, tol = 1e-5, x0 = init_sig)

    def __call__(self):
        cg_res = np.asarray(self.apply_cg()[0])
        print(cg_res)
        final_alm = hp.almxfl(cg_res, np.sqrt(self.C_ell))
        return final_alm

#CG returns the solution vector C^-1/2s, but what I need for the next sampling step is actually just s, so let us incorporate that into the class here :) 
















    







    