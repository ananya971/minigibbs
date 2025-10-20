import nifty.re as jft
import jax.numpy as jnp 
import numpy as np 
import healpy as hp 
import jax
from utils import load_config
from nifty.re import cg
import jaxbind.contrib.jaxducc0 as jaxducc0
from jax.tree_util import tree_structure

c= load_config()

key = jax.random.PRNGKey(c['seed'])

class CG:
    def __init__(self, c, data, noise, C_ell):
        self.c = c
        self.data = data
        self.noise = noise
        self.Ninv = noise**-1
        self.C_ell = C_ell
        self.npix = hp.nside2npix(self.c['nside'])
        self.ps = self.C_ell.shape[0]
        self.res = hp.nside2resol(nside = self.c['nside']) #[rad]
        self.beam = hp.gauss_beam(fwhm = self.res, lmax = 2 * self.c['nside'])
        self.pwf = hp.pixwin(nside = self.c['nside'], lmax = 2 * self.c['nside'])
        self.Al = self.beam*self.pwf
    
    def apply_A(self, s, n_freq):
        s_beam = hp.almxfl(alm = s, fl = self.Al)
        spix_beam = hp.alm2map(s_beam, nside = self.c['nside'], lmax = 2 * self.c['nside'])
        return np.tile(spix_beam, n_freq)
    
    def adjoint_alm2map(self, x):
        sht_T = jax.linear_transpose(jaxducc0.get_healpix_sht(nside = self.c['nside'], lmax = 2 * self.c['nside'], mmax = 2 * self.c['nside'], spin = 0, nthreads=1), jax.ShapeDtypeStruct(shape = (1, hp.Alm.getsize(lmax = 2 * self.c['nside'])), dtype=jnp.complex128))
        print("Output structure:", tree_structure(x))
        return jnp.conj(sht_T(x))

    def apply_A_transpose(self, stacked, n_freq):
        s_summed = stacked.reshape(n_freq, self.npix).sum(axis = 0)
        spix_summed = self.adjoint_alm2map(jnp.asarray(s_summed))
        s_beamed = hp.almxfl(alm = spix_summed, fl = self.Al)
        return s_beamed
    

    def apply_mat(self,x):
        '''Apply matrix, finish doc'''
        x = np.asarray(x)
        alm_cl = hp.almxfl(alm = x, fl = np.sqrt(self.C_ell)) # S^(1/2)x
        A_pix_alm = self.apply_A(alm_cl, self.c['nfreqs'])
        NA_pix = self.Ninv * A_pix_alm
        ATNA_pix = self.apply_A_transpose(NA_pix, self.c['nfreqs'])
        CATNA_alm = hp.almxfl(ATNA_pix, fl = np.sqrt(self.C_ell))
        return CATNA_alm + x
    
    def rhs(self):
        term_1a = self.Ninv * self.data
        term_1b = self.apply_A_transpose(term_1a, self.c['nfreqs'])
        term_1 = hp.almxfl(term_1b, fl = np.sqrt(self.C_ell)) # C^1/2 AN^-1d

        omega1 = jft.random_like(key, jax.ShapeDtypeStruct(shape = self.data.shape, dtype = jnp.float64))

        term_2a = np.sqrt(self.Ninv) * np.asarray(omega1)
        term_2b = self.apply_A_transpose(term_2a, self.c['nfreqs'])
        term_2 = hp.almxfl(term_2b, fl = np.sqrt(self.C_ell))

        omega0 = jft.random_like(key, jax.ShapeDtypeStruct(shape = (int((self.ps * (self.ps -1))/2 + (self.ps)) ,), dtype = jnp.complex64))
        return term_1 + term_2 + np.asarray(omega0)
    
    def apply_cg(self):
        rhs = self.rhs()
        init_sig = hp.synalm(self.C_ell, lmax = 2 * self.c['nside'])
        return cg(self.apply_mat, rhs, miniter = 500, x0 = init_sig )#name = 'CG')

    def __call__(self):
        cg_res = np.asarray(self.apply_cg())
        final_alm = hp.almxfl(cg_res, np.sqrt(self.C_ell))
        return final_alm

















    







    