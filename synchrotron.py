import nifty.re as jft
import healpy as hp 
import jax.numpy as jnp 
from utils import load_config
from astropy.io import fits
import numpy as np
from scipy import interpolate
import jax

class Synchrotron:
    def __init__(self, c, file, Ninv, data):
        self.c = c 
        self.nus = np.asarray(c['freqs'])
        self.file = file
        self.alpha = self.c['components']['synchrotron']['alpha']
        self.Ninv = Ninv
        self.data = data

    def read_maps(self):
        synch_map  = hp.read_map(self.file ,h = True, field = 1) #[K_RJ]
        EM_s = synch_map[0]/1e6
        nu_ref = dict(synch_map[1])['NU_REF'].split(' ')
        nu_ref = float(nu_ref[0]) * 1e6 if nu_ref[1]=='MHz' else float(nu_ref[0]) * 1e9
        return EM_s, nu_ref

    
    def synch_comm(self):
        A, nu_ref = self.read_maps()
        hdul = fits.open(self.file)
        nus = [np.asarray(hdul[2].data[i])[0]*1e9 for i in range(78)]  # [Hz]
        nus = [int(i) for i in nus]
        i = [np.asarray(hdul[2].data)[i][1] for i in range(78)]
        twerp = interpolate.interp1d(nus, i) 
        s = A.reshape(1, A.shape[0]) * ((nu_ref/self.nus)**2).reshape(self.nus.shape[0],1) * (twerp(self.nus/self.alpha)/twerp(nu_ref/self.alpha)).reshape(self.nus.shape[0],1)

        return s 
    
    def get_rhs(self):
        omega0 = jax.random.normal(key = jax.random.PRNGKey(self.c['seed']), shape = (self.npix,), dtype = jnp.float64)
        term_1 = self.synch_comm().T * jnp.sqrt(self.Ninv) * omega0
        term_2a = self.Ninv * self.data
        term_2 = self.synch_comm().T * term_2a

        return term_1, term_2
    
    def __call__(self):
        if self.c['components']['synchrotron']['flavour'] == 'temp':
            return self.synch_comm(), self.get_rhs()
        elif self.c['components']['synchrotron']['flavour'] == 'tempspec':
            raise NotImplementedError    





