import jax 
import healpy as hp 
import camb
import nifty.re as jft
from cmb import CMB
from utils import load_config
from sample import Cl_given_s, gibbs
import numpy as np
from utils import Dl_to_Cl
from cg import CG


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

nvar = 1e-4

noise_truth_1 = np.ones(hp.nside2npix(c['nside']))*nvar
noise_truth_2 = np.ones(hp.nside2npix(c['nside']))*nvar
data_1 = cmb_field + noise_truth_1

# check power spectra of data and ground truth, noise only dominates at ell~80

data_2 = cmb_field + noise_truth_2

all_data = np.append(data_1, data_2)
all_noise = np.append(noise_truth_1, noise_truth_2)

assert all_data.shape == all_noise.shape, "Data and noise shapes do not match!"

# Sample from C_ell

cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])


result_ps, result_alm = gibbs(iter = 5, init_ps= TTCl, data = all_data, noise = all_noise, nside = c['nside']) 


# Checking power spectra of data vs. cmb
# import matplotlib.pyplot as plt 

# data1_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(data_1), lmax = 2 * c['nside']), lmax = 2*c['nside'])

# plt.plot(jnp.arange(data1_cl.shape[0]), data1_cl)
# plt.xscale('log')

# cmb_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside']), lmax = 2*c['nside'])
# plt.plot(jnp.arange(cmb_cl.shape[0]), cmb_cl)
# plt.xscale('log')

# hp.mollview(hp.alm2map(result_alm, nside = c['nside'], lmax = 2 * c['nside'])*1e6, norm = 'hist')











