import jax 
import healpy as hp 
import camb
import nifty.re as jft
from cmb import CMB
from utils import load_config
from sample import gibbs
import numpy as np
from utils import Dl_to_Cl
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

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
TTCl = Dl_to_Cl(power_spec[:, 0]) 

seed = c['seed']
key = jax.random.PRNGKey(seed)
cmb_model = CMB(c=c)

input = jft.random_like(key, cmb_model.domain)

cmb_field = cmb_model(input)[0][0] # Only take the CMB I signal for one frequency band. 

key, subkey = jax.random.split(key, num= 2) 

nstd = 1e-4

noise_truth_1 = (jax.random.normal(key, shape = cmb_field.shape, dtype = jnp.float64) * nstd) # Z = (X-mu)/sigma where Z is standard normally distributed
noise_truth_2 = (jax.random.normal(subkey, shape = cmb_field.shape, dtype = jnp.float64) * nstd)

cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])

data_1 = hp.alm2map(hp.almxfl(cmb_alms, fl = hp.gauss_beam(2 * hp.nside2resol(c['nside']))), nside = c['nside'], lmax = 2 * c['nside']) + noise_truth_1 # d_1 = A(s) + n_1

data_2 = hp.alm2map(hp.almxfl(cmb_alms, fl = hp.gauss_beam(2 * hp.nside2resol(nside = c['nside']))), nside = c['nside'], lmax = 2 * c['nside']) + noise_truth_2 # d_2 = A(s) + n_2

all_data = np.append(data_1, data_2)

cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])

result_ps, result_alm = gibbs(iter = 10, init_ps= TTCl, data = all_data, noise =nstd**2, nside = c['nside']) 
