import healpy as hp 
import jax.numpy as jnp 
import numpy as np
import jax


class FreeFree:
    def __init__(self, c, file, Ninv):
        self.c = c 
        self.nus = np.asarray(c['freqs'])
        self.file = file 
        self.npix = hp.nside2npix(self.c['nside'])
        self.Ninv = Ninv
        self.nside = self.c['nside']
        
    def read_maps(self):
        EM_ff = jnp.asarray(hp.ud_grade(hp.read_map(self.file, field= 1), dtype = jnp.float64, nside_out=self.nside))
        tempy = jnp.asarray(hp.ud_grade(hp.read_map(self.file, field= 4), dtype = jnp.float64, nside_out=self.nside))
        return EM_ff, tempy

    def freefree_comm(self):
        # T_eff = 7000.0 # K
        EMff, temp = self.read_maps()
        nu_GHz = jnp.asarray(self.nus) * 1e-9
        g_ff = jnp.log(jnp.exp(5.960 - jnp.sqrt(3)/(jnp.pi * jnp.log(nu_GHz.reshape(nu_GHz.shape[0],1)* (1e-4 * temp.reshape(1,temp.shape[0]))**(-3/2)))) + jnp.e)
        tau = 0.05468 * (nu_GHz.reshape(nu_GHz.shape[0],1))**-2 * temp.reshape(1,temp.shape[0])**(-3/2) * g_ff * EMff
        free_free = temp * (-1 * jnp.expm1(-tau))
        return free_free
    
    def get_rhs(self, data):
        omega0 = jax.random.normal(key = jax.random.PRNGKey(self.c['seed']), shape = (len(self.nus), self.npix), dtype = jnp.float64)
        term_1 = jnp.sqrt(self.Ninv) * jnp.einsum('ij,ji', self.freefree_comm().T, omega0)
        term_2a = self.Ninv * data
        term_2 = jnp.einsum('ij,ji', self.freefree_comm().T, term_2a)

        return term_1 + term_2

    def __call__(self, data):
        if self.c['components']['freefree']['flavour'] == 'temp':
            return  self.get_rhs(data)
        elif self.c['components']['free']['flavour'] == 'tempspec':
            raise NotImplementedError




