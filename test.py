import jax 
from jax import numpy as jnp 
import scipy 
import healpy as hp 
import camb
import nifty.re as jft
from cmb import CMB
from utils import load_config
from sample import Cl_given_s, gibbs
import numpy as np
from utils import Dl_to_Cl
from cg import CG
from sample import get_sig_ps_healpy


from nifty.re import cg
from functools import partial

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

c = load_config()
seed = c['seed']

key = jax.random.PRNGKey(seed)
cmb_model = CMB(c=c)

input = jft.random_like(key, cmb_model.domain)

cmb_field = cmb_model(input)[0][0]

key, subkey = jax.random.split(key, num= 2)

noise_1 = lambda x: 1e-6 * x # Try also with one zero less

noise_truth_1 = ((
    (noise_1(jft.ones_like(cmb_model.target))) ** 0.5
) * np.abs(jft.random_like(key, cmb_model.target)))[0][0]

noise_truth_2 = ((
    (noise_1(jft.ones_like(cmb_model.target))) ** 0.5
) * np.abs(jft.random_like(subkey, cmb_model.target)))[0][0]

data_1 = cmb_field + noise_truth_1

data_2 = cmb_field + noise_truth_2

all_data = np.append(data_1, data_2)
all_noise = np.append(noise_truth_1, noise_truth_2)

assert all_data.shape == all_noise.shape, "Data and noise shapes do not match!"

cgggg = CG(c = c, data = all_data, noise = all_noise, C_ell = TTCl)

hp.mollview(hp.alm2map(cgggg(), nside = c['nside'], lmax = 2 * c['nside'])*1e6)


# Sample from C_ell

cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])


result_ps, result_alm = gibbs(iter = 20, init_ps= TTCl, data = all_data, noise = all_noise, nside = c['nside']) 















